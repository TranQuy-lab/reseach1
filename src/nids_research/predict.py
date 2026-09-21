"""Predict offline on an unlabeled Parquet graph using a saved run."""
import argparse
from pathlib import Path
import pandas as pd
from .experiment import predict_artifact


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        p.error('Refusing to overwrite output')
    frame = pd.read_parquet(a.input)
    probabilities, classes = predict_artifact(a.run, frame)
    result = pd.DataFrame({'prediction': [classes[i] for i in probabilities.argmax(1)]})
    if '_flow_id' in frame:
        result.insert(0, 'flow_id', frame._flow_id.to_numpy())
    for i,c in enumerate(classes):
        result[f'p_{c}'] = probabilities[:,i]
    result.to_parquet(a.output, index=False)


if __name__ == '__main__':
    main()
