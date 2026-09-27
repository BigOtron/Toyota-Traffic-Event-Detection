## Technical report

### What we built

A pipeline that watches the fixed camera and reports **jaywalking** (a pedestrian on the carriageway outside a crossing) as time segments.

1. **Detection + tracking.** YOLOv8s (COCO, no fine-tuning) + ByteTrack on every 3rd frame (~10 fps). Classes: person, bicycle, car, motorcycle, bus, truck.
2. **Camera alignment.** The camera is not perfectly fixed: between recordings it moves by up to ~110 px in 4K. We match each video to one reference frame with SIFT features on the static lower part of the image (road markings, curbs) and a RANSAC homography. We try 5 frames (0–12 s) and keep the best. All zones live in the reference frame.
3. **Scene zones** (drawn by hand once, `configs/scene.json`): carriageway, 3 crosswalks, 5 islands, the far side of the crossings (bus stop / waiting area).
4. **Jaywalking rule.** A person point counts if it is ≥80 px inside the road, ≥100 px outside every crosswalk, not on an island, and not on a bicycle or motorcycle. A track must stay there for ≥1 s and move ≥250 px at walking speed (so people waiting at a curb and tracker ID jumps do not count). Runs are padded by 1.5 s, merged if closer than 4 s, and kept if longer than 2 s. A group crossing together is one event.

**Learned:** only the detector. **Rule-based:** everything else.

### Results on the sample videos (our own labels, 13 events)

| | tIoU 0.3 | tIoU 0.5 | tIoU 0.7 |
|---|---|---|---|
| F1 | 0.714 | 0.714 | 0.643 |
| TP / FP / FN | 10 / 5 / 3 | 10 / 5 / 3 | 9 / 6 / 4 |

**Score A = 0.690** (full run with the organizers' `run_submission.py`). Runtime ≈ 1.0–1.6× video length on a laptop GPU (RTX 5050), budget is 3×.

| Video | Light | TP / FP / FN @0.5 |
|---|---|---|
| C3896 | day | 2 / 0 / 0 |
| C3897 | day | 4 / 3 / 0 |
| C3902 | day, camera shifted ~110 px | 3 / 2 / 1 |
| C3905 | dusk | 1 / 0 / 2 |

The thresholds were tuned on these same 13 events, so the numbers are optimistic.

### What worked

- **Aligning every video to one reference frame.** Without it, zones drawn on one video land on the wrong place in another (C3902 is shifted by ~110 px).
- **Requiring movement, not just position.** Most false alarms were people standing at the curb edge or tracker boxes jumping between people.
- **Saving tracks once and tuning rules on CSVs.** A rule change takes seconds to evaluate instead of a 20-minute video run.

### What did not work / failure cases

- **Dusk (C3905):** fewer and shorter person tracks, 2 of 3 events missed.
- **False positives** come from pedestrians who step just outside the painted crosswalk, and from a few crossings we may have missed in our own labels (C3897 238–245 s).
- **A bug we found and fixed:** reusing one YOLO model for several videos made Ultralytics register the tracker callbacks again for every video, so from the 2nd video on tracks broke. Score went from 0.29 to 0.69 after creating a fresh model per video.
- **Only one class.** The other 13 classes are not predicted. Predicting a class that is not in the test set would lower the macro F1, so we only ship what we measured.
- **Part B** (accident anticipation) is the default zero-risk estimator.

### What we would do next

1. `failure_to_yield` and `stopped_vehicle`: they reuse the same tracks and zones.
2. A larger detector (YOLOv8m) or a night-tuned model for dusk videos.
3. Part B: time-to-collision between tracked vehicles and pedestrians.
4. More labelled data: our dev set has only 13 events.
