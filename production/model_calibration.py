#This library comes from https://github.com/p-lambda/verified_calibration
#The library is based on a paper from the same author https://arxiv.org/pdf/1909.10155
import calibration as cal
import production.utils as u
from sklearn.calibration import calibration_curve, CalibrationDisplay, CalibratedClassifierCV
from sklearn import metrics as m
from joblib import dump, load

def expected_calibration_error(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece

def calculate_calibration_errors(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)
    if len(y_test.shape) != 1:
        y_test = y_test.values.reshape(1,-1).flatten()
    calibration_error = cal.get_calibration_error(y_pred_proba, y_test)
    prob_pos = y_pred_proba[:, 1]
    CalibrationDisplay.from_predictions(y_test, prob_pos, n_bins=10)
    brier = m.brier_score_loss(y_test, prob_pos)
    return calibration_error, brier

def run_calibration_process(uncalibrated_model, X_train, y_train, X_test, y_test):
    initial_calibration_error, initial_brier = calculate_calibration_errors(uncalibrated_model, X_test, y_test)
    print(f"Initial calibration error (perfectly calibrated = 0): {initial_calibration_error:0.3f}")
    print(f"Initial Brier score (lower = better): {initial_brier}")

    calibrated_model = CalibratedClassifierCV(uncalibrated_model, method='isotonic', cv=5)
    #Calculates isotonic regression on out-of-fold data so it should be safe to use training data here again
    calibrated_model.fit(X_train, y_train)

    print('=====================Post Calibration analysis=====================')
    final_metrics = {}
    
    final_calibration_error, final_brier = calculate_calibration_errors(calibrated_model, X_test, y_test)
    final_metrics['final_calibration_error'] = final_calibration_error
    final_metrics['brier_calibrated'] = final_brier
    print(f"Final calibration error (perfectly calibrated = 0): {final_metrics['final_calibration_error']:0.3f}")
    print(f"Final Brier score (lower = better): {final_metrics['brier_calibrated']:0.3f}")

    y_pred_calibrated = calibrated_model.predict(X_test)
    final_metrics['confusion_matrix'] = m.confusion_matrix(y_test,y_pred_calibrated)
    final_metrics['precision'] = m.precision_score(y_test, y_pred_calibrated)
    final_metrics['recall'] = m.recall_score(y_test, y_pred_calibrated)

    return calibrated_model, final_metrics

def save_calibrated_model(calibrated_model):
    model_directory = u.get_directory('models')
    dump(calibrated_model, f'{model_directory}/calibrated_model.joblib')
    return True