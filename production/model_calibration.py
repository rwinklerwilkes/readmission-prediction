import calibration as cal

#TODO: Convert to functional

calibration_error = cal.get_calibration_error(clf.predict_proba(X_test_to_use), y_test)

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

expected_calibration_error(clf.predict_proba(X_test_to_use), y_test,10)

from sklearn.calibration import calibration_curve, CalibrationDisplay

prob_pos = clf.predict_proba(X_test_to_use)[:, 1]
true = y_test
CalibrationDisplay.from_predictions(true, prob_pos, n_bins=10)

from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_test, prob_pos)
print(brier)

from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
calibrated_model.fit(X_train_to_use, y_train)

prob_pos = calibrated_model.predict_proba(X_test_to_use)[:, 1]
true = y_test
CalibrationDisplay.from_predictions(true, prob_pos, n_bins=10)

from sklearn import metrics as m

y_pred_calibrated = calibrated_model.predict(X_test_to_use)
print(m.confusion_matrix(y_test,y_pred_calibrated))
print(m.precision_score(y_test, y_pred_calibrated))
print(m.recall_score(y_test, y_pred_calibrated))

from joblib import dump, load

dump(calibrated_model, 'models/calibrated_model.joblib')