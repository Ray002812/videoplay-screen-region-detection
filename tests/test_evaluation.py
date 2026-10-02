from videoplay.evaluation import normalize_coco_gt, xyxy_to_coco_bbox


def test_xyxy_to_coco_bbox():
    assert xyxy_to_coco_bbox(10, 20, 40, 50) == [10.0, 20.0, 30.0, 30.0]


def test_normalize_adds_images():
    data = {"annotations": [{"image_id": "a-1", "bbox": [0, 0, 1, 1], "category_id": 1}]}
    out = normalize_coco_gt(data)
    assert "images" in out
    assert out["images"][0]["id"] == "a-1"
    assert "info" in out
