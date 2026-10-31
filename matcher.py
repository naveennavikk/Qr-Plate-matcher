import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class HybridImageMatcher:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(self.device)
        self.orb = cv2.ORB_create(nfeatures=8000, scaleFactor=1.15, nlevels=10, edgeThreshold=8, fastThreshold=7)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _regions(self, image, prompt):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.18, text_threshold=0.15,
            target_sizes=[image.size[::-1]],
        )[0]
        return results["boxes"].cpu().numpy()

    def _match_region(self, ref_gray, scene_gray, offset_x=0, offset_y=0, min_inliers=8):
        kp1, des1 = self.orb.detectAndCompute(ref_gray, None)
        kp2, des2 = self.orb.detectAndCompute(scene_gray, None)
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return None
        pairs = self.matcher.knnMatch(des1, des2, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.72 * n.distance]
        if len(good) < min_inliers:
            return None
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        if homography is None or inliers < min_inliers:
            return None
        h, w = ref_gray.shape
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        x1, y1 = np.floor(projected.min(axis=0)).astype(int)
        x2, y2 = np.ceil(projected.max(axis=0)).astype(int)
        return inliers, [int(x1 + offset_x), int(y1 + offset_y), int(x2 + offset_x), int(y2 + offset_y)]

    def _template_search(self, ref_gray, scene_gray):
        """Find a small reference using normalized correlation at multiple scales."""
        rh, rw = ref_gray.shape
        sh, sw = scene_gray.shape
        if rh < 12 or rw < 12 or sh < 12 or sw < 12:
            return None
        # Matching the grayscale and edge images helps with exposure and color changes.
        # Remove low-frequency illumination/background differences. This matters when
        # the close-up contains wall around the plate, but the plate is tiny at distance.
        ref_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ref_gray)
        scene_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(scene_gray)
        ref_edge = cv2.Canny(cv2.GaussianBlur(ref_gray, (3, 3), 0), 35, 110)
        scene_edge = cv2.Canny(cv2.GaussianBlur(scene_gray, (3, 3), 0), 45, 130)
        best = None
        # Use fewer scales; resizing the property image below also limits per-scale work.
        for scale in np.geomspace(0.10, 1.8, 21):
            tw, th = int(round(rw * scale)), int(round(rh * scale))
            if tw < 12 or th < 12 or tw > sw or th > sh:
                continue
            template = cv2.resize(ref_gray, (tw, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            templ_edge = cv2.resize(ref_edge, (tw, th), interpolation=cv2.INTER_NEAREST)
            for image, templ in ((scene_gray, template), (scene_edge, templ_edge)):
                if float(templ.std()) < 2 or float(image.std()) < 2:
                    continue
                scores = cv2.matchTemplate(image, templ, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(scores)
                if best is None or score > best[0]:
                    best = (float(score), loc[0], loc[1], tw, th)
        if best is None:
            return None
        score, x, y, tw, th = best
        return score, [x, y, x + tw, y + th]

    def verify_presence(self, ref_img, property_img, target_prompt="plate . sign . house number ."):
        if not isinstance(ref_img, Image.Image):
            ref_img = Image.open(ref_img)
        if not isinstance(property_img, Image.Image):
            property_img = Image.open(property_img)
        ref_img = ref_img.convert("RGB")
        property_img = property_img.convert("RGB")
        # Keep matching fast for large camera photos. Scale final coordinates back
        # to the uploaded property's resolution before returning them.
        max_side = 1400
        property_scale = min(1.0, max_side / max(property_img.size))
        if property_scale < 1.0:
            match_property = property_img.resize(
                (round(property_img.width * property_scale), round(property_img.height * property_scale)),
                Image.Resampling.LANCZOS,
            )
        else:
            match_property = property_img
        ref = np.asarray(ref_img)
        prop = np.asarray(match_property)
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
        prop_gray = cv2.cvtColor(prop, cv2.COLOR_RGB2GRAY)
        height, width = prop_gray.shape
        # Template matching provides the fast path. Grounding DINO runs only if that
        # pass misses and a non-empty region hint was supplied.
        # Focus comparison on the likely object instead of the surrounding wall. A
        # plate often occupies only a small fraction of the reference upload.
        ref_h, ref_w = ref_gray.shape
        plate_crop = ref_gray
        plate_box = None
        if ref_h > 100 and ref_w > 100:
            # Bright, near-neutral plate pixels stand out from the green wall. Use a
            # broad threshold and retain a margin so QR/printed details stay included.
            hsv = cv2.cvtColor(ref, cv2.COLOR_RGB2HSV)
            white = cv2.inRange(hsv, np.array([0, 0, 105]), np.array([180, 150, 255]))
            white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > (ref_w * ref_h * 0.002) and w > 18 and h > 12:
                    margin_x, margin_y = max(3, int(w * .08)), max(3, int(h * .08))
                    x1, y1 = max(0, x-margin_x), max(0, y-margin_y)
                    x2, y2 = min(ref_w, x+w+margin_x), min(ref_h, y+h+margin_y)
                    plate_crop = ref_gray[y1:y2, x1:x2]
                    plate_box = (x1, y1, x2, y2)
        # Search the property on a coarse-to-fine grid of overlapping crops. This
        # keeps the returned box plate-sized and prevents a homography from expanding
        # to the full photo when only a few features match.
        template_match = None
        crop_h, crop_w = plate_crop.shape
        expected_scales = np.geomspace(0.08, 1.6, 18)
        for scale in expected_scales:
            tw, th = round(crop_w * scale), round(crop_h * scale)
            if tw < 16 or th < 12 or tw > width or th > height:
                continue
            step_x, step_y = max(1, tw // 2), max(1, th // 2)
            for yy in range(0, max(1, height - th + 1), step_y):
                for xx in range(0, max(1, width - tw + 1), step_x):
                    patch = prop_gray[yy:min(height, yy+th), xx:min(width, xx+tw)]
                    if patch.shape[0] < th or patch.shape[1] < tw:
                        continue
                    # At each expected scale the local patch is only slightly larger
                    # than the template, so a single-scale match is sufficient.
                    template = cv2.resize(
                        plate_crop, (tw, th),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
                    )
                    template_edge = cv2.Canny(cv2.GaussianBlur(template, (3, 3), 0), 35, 110)
                    patch_edge = cv2.Canny(cv2.GaussianBlur(patch, (3, 3), 0), 45, 130)
                    if template_edge.std() < 2 or patch_edge.std() < 2:
                        continue
                    scores = cv2.matchTemplate(patch_edge, template_edge, cv2.TM_CCOEFF_NORMED)
                    _, score, _, loc = cv2.minMaxLoc(scores)
                    if template_match is None or score > template_match[0]:
                        template_match = (float(score), [xx + loc[0], yy + loc[1], xx + loc[0] + tw, yy + loc[1] + th])
        if template_match and template_match[0] >= 0.43:
            score, bbox = template_match
            if plate_box:
                # Report the whole plate area, not only the thresholded white surface.
                scale_x = (bbox[2] - bbox[0]) / max(1, plate_crop.shape[1])
                scale_y = (bbox[3] - bbox[1]) / max(1, plate_crop.shape[0])
                bbox = [
                    round(bbox[0] - plate_box[0] * scale_x),
                    round(bbox[1] - plate_box[1] * scale_y),
                    round(bbox[2] + (ref_w - plate_box[2]) * scale_x),
                    round(bbox[3] + (ref_h - plate_box[3]) * scale_y),
                ]
            x1, y1, x2, y2 = bbox
            return {
                "contains_reference": True,
                "bbox": [
                    max(0, round(x1 / property_scale)), max(0, round(y1 / property_scale)),
                    min(property_img.width, round(x2 / property_scale)),
                    min(property_img.height, round(y2 / property_scale)),
                ],
                "match_score": score,
                "detection_method": "Localized plate template matching",
            }
        if target_prompt.strip():
            try:
                boxes = self._regions(match_property, target_prompt)
                for box in boxes:
                    x1, y1, x2, y2 = np.rint(box).astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    found = self._template_search(plate_crop, prop_gray[y1:y2, x1:x2])
                    if found and found[0] >= 0.43:
                        score, local = found
                        bx1, by1, bx2, by2 = local
                        return {
                            "contains_reference": True,
                            "bbox": [
                                max(0, round((x1 + bx1) / property_scale)),
                                max(0, round((y1 + by1) / property_scale)),
                                min(property_img.width, round((x1 + bx2) / property_scale)),
                                min(property_img.height, round((y1 + by2) / property_scale)),
                            ],
                            "match_score": score,
                            "detection_method": "Grounding DINO fallback + plate template matching",
                        }
            except Exception:
                # Keep the fast path usable when model downloads/inference are unavailable.
                pass
        return {"contains_reference": False, "best_template_score": template_match[0] if template_match else None}
