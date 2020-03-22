import numpy as np
import torch


def predict_daily_cases(model, X_test, y_test, days_to_predict, seq_len, scaler):
    with torch.no_grad():
        test_seq = X_test[:1]
        preds = []
        for _ in range(days_to_predict):
            y_test_pred = model(test_seq)
            pred = torch.flatten(y_test_pred).item()
            preds.append(pred)
            new_seq = test_seq.numpy().flatten()
            new_seq = np.append(new_seq, [pred])
            new_seq = new_seq[1:]
            test_seq = torch.as_tensor(new_seq).view(1, seq_len, 1).float()

    predicted_cases = scaler.inverse_transform(
        np.expand_dims(preds, axis=0)).flatten()

    true_cases = scaler.inverse_transform(
        np.expand_dims(y_test.flatten().numpy(), axis=0)).flatten()

    return predicted_cases, true_cases
