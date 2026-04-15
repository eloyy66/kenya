# Mostacho TFLite 2.2 - Evaluation Report

- Generated at (UTC): `2026-04-09T00:06:34Z`

## drowsiness_int8
- Accuracy: `0.9029`
- Precision macro: `0.9366`
- Recall macro: `0.8112`
- F1 macro: `0.8480`
- Samples: `5004`

## eye_float32
- Accuracy: `0.9600`
- Precision macro: `0.9629`
- Recall macro: `0.9600`
- F1 macro: `0.9599`
- Samples: `4000`
- ROC AUC (open): `0.9996`
- Best threshold by F1: `0.10` (F1 `0.9907`, P `0.9970`, R `0.9845`)

## distraction_float32
- Accuracy: `0.7508`
- Precision macro: `0.7702`
- Recall macro: `0.7836`
- F1 macro: `0.7498`
- Samples: `1232`
- ROC AUC (distracted): `0.9165`
- Best threshold by F1: `0.55` (F1 `0.8037`, P `0.7667`, R `0.8445`)
