
# 🚨 STRESS TEST INSERTION CODE BLOCK for recursive_forecast()
# Replace inside the for-loop in recursive_forecast() after reading `row`
# Only activate if stress_test is True

if stress_test:
    # Inject noise into lag features
    if not pd.isnull(lag_24h): lag_24h *= np.random.normal(1, 0.05)
    if not pd.isnull(lag_168h): lag_168h *= np.random.normal(1, 0.05)
    # Simulate cold weather shock
    row["temp_C"] -= np.random.choice([0, 2, 4])
