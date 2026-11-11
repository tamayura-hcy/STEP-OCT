# Data

The three cohorts are public and are **not** redistributed with this repository.
Download them from the original providers and lay them out as below; the loaders in
`step/data.py` expect an `ImageFolder` tree with one directory per class.

```
data/
  BOE/{train,val,test}/{AMD,DME,NORMAL}/*.jpeg
  TMI/{train,val,test}/{AMD,DME,NORMAL}/*.jpeg
  CELL/{train,val,test}/{AMD,DME,NORMAL}/*.jpeg
```

| ID | Cohort | Scanner | Classes | Images | Reference |
|----|--------|---------|---------|--------|-----------|
| A | 2014\_BOE\_Srinivasan | Spectralis SD-OCT (Heidelberg Engineering) | AMD, DME, NORMAL | 3231 (723 / 1101 / 1407), 45 patients | Srinivasan *et al.*, Biomed. Opt. Express 5(10):3568–3577, 2014 |
| B | Macular Dataset–Heidelberg | Heidelberg | AMD, DME, NORMAL | 3755 (1318 / 852 / 1585), 148 subjects; 918 test images | Noor Eye Hospital, Tehran |
| C | OCT2017 / ZhangLabData | — | AMD, DME, NORMAL | 2226 (742 per class); 558 test images | Kermany *et al.*, Cell 172(5):1122–1131, 2018 |

The three transfers are A→B, A→C and B→C. A→B and A→C share the source model trained
on A; A→C and B→C share the target cohort C.

## What the released files are, and are not

Only 8-bit JPEG exports are available from the providers, with no EXIF, no calibration
and no acquisition metadata. The pixel values are therefore a *rendered* intensity and
not a calibrated one, which is exactly why the paper treats the difference between
cohorts as a difference of local contrast rather than of absolute brightness, and why
it measures the device signature on the released pixels.

A note for anyone reproducing the numbers: within a cohort directory the released files
are not a perfectly uniform acquisition set (the archives mix a few sub-formats). The
paper's protocol uses the official train/val/test split labels of each provider and
scores every released test image, so the numbers are comparable across methods but
should not be read as a controlled "one scanner, one protocol" study.
