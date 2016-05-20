# IMT4904 MasterThesis Source code: Feature detectors

Contains file with singular feature detectors. E.g. "feature_detector_10000_homework.csv" indicates that this file 
contains 20.000 rows (10.000 good and bad questions), and that it contains the phrases 'has_homework' and 'has_assignment'.

Filename explanation:
- Numeric value: Amount of rows retrieved from database * 2 (good and bad)
- _unprocessed: Based on the unprocessed data set (contains only named feature)
- _type: Type (e.g. 'has_codeblock') indicates the type of feature that was used to create this file
