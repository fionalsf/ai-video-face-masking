# V3 lazy-review workflow

V3 keeps the V2 mask-timeline contract, but changes review package export to
avoid generating every contact sheet before review starts.

## What changed

- `export_mask_review_pack()` defaults to lazy preview mode.
- Pipeline/prepare step writes only:
  - `mask_timeline.json`
  - `review/pending_events.json`
  - `review/suppressed_events.json`
  - `review/confirmed_events.json`
- `review/event_contact_sheet/*.jpg` is generated on demand by `review_ui.py`
  when the reviewer opens an event for the first time.
- Generated contact sheets are cached on disk, so revisiting an event is fast.

This removes the old batch contact-sheet generation cost from "video processed
to review ready". On the 37-minute test output, rebuilding the review package
from existing `face_events.json` dropped from tens of minutes to about 1-2
seconds.

## Commands

Build a lazy review pack from an existing output directory:

```powershell
cd "D:\work\制造业视频数据\new_da_ma"
python prepare_mask_review.py --output-dir "D:\path\to\output_dir"
```

Open the review UI:

```powershell
cd "D:\work\制造业视频数据\new_da_ma"
streamlit run review_ui.py -- --review-dir "D:\path\to\output_dir\review"
```

Force the old eager preview behavior only when needed:

```powershell
python prepare_mask_review.py --output-dir "D:\path\to\output_dir" --eager-previews
```

## Expected behavior

The review page may show a short "Generating preview..." spinner the first time
an event is opened. That cost is paid per event instead of blocking the whole
review package export.

