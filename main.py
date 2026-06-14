from production import data_preprocessing as d
from production import data_validations as v
from production import feature_engineering as f
from production import model as m
from production import model_calibration as mc


def run_validations(df):
    v.validate_against_schema(df)

    new_stats = v.calculate_psi_statistics(df)
    old_stats = v.load_initial_statistics()

    for colname in new_stats.keys():
        _, psi = v.population_stability_index(
            old_stats[colname], new_stats[colname], colname, pregrouped=True
        )
        if psi > 0.1:
            raise ValueError(
                f"Population for {colname} has shifted, check before running model"
            )


def feature_engineering_steps(df):
    # Save some data that I'm likely to need later for streamlit
    diag = f.count_all_diag(df)
    icd9 = f.load_icd9()
    diag = f.join_diag_to_icd9(diag, icd9)
    f.stash_data(df, diag)

    df = f.create_dummies(df)
    df = f.parse_weight_to_number(df)
    df = f.add_prior_admissions(df)

    alive, expired = f.filter_for_still_alive(df)
    # Validate that nobody who expired was readmitted later
    assert expired[expired["readmitted_dummy"] == 1].count()["encounter_id"] == 0

    alive = f.compress_medicine_columns(alive)
    icd9_features = f.icd9_featurization(icd9)
    alive = f.add_icd9_features(alive, icd9_features)
    alive = f.add_random_feature(alive)
    final_feature_df = f.select_final_features(alive)
    return final_feature_df


def model_training(feature_df):
    X, y = m.split_X_and_y(feature_df)
    X_train, X_test, y_train, y_test = m.train_test_split(X, y)
    model = m.train_model(X_train, y_train)
    output_metrics = m.calculate_metrics(model, X_test, y_test)
    champion_model = m.compare_current_to_champion(model, output_metrics, "precision")
    return champion_model, X_train, X_test, y_train, y_test


def model_calibration(champion_model, X_train, y_train, X_test, y_test):
    calibrated_model, calibrated_metrics = mc.run_calibration_process(
        champion_model, X_train, y_train, X_test, y_test
    )
    mc.save_calibrated_model(calibrated_model)
    return calibrated_model, calibrated_metrics


def main():
    df = d.load_file()
    run_validations(df)
    feature_df = feature_engineering_steps(df)
    champion_model, X_train, X_test, y_train, y_test = model_training(feature_df)
    calibrated_model, calibrated_metrics = model_calibration(
        champion_model, X_train, X_test, y_train, y_test
    )
    return calibrated_model


if __name__ == "__main__":
    calibrated_model = main()
