from mlkit.config.data_process import FeatureCard, FeatureType, FeatureClass, Scope

feature_lib = {
    # ONLINE FEATURES (current transaction)
    "purchase_date": FeatureCard(
        name="purchase_date",
        description="Date of the purchase",
        feature_type=FeatureType.DATETIME,
        feature_class=FeatureClass.BASIC,
        scope=Scope.ONLINE,
    ),
    "age": FeatureCard(
        name="age",
        description="Age of the customer",
        feature_type=FeatureType.FLOAT64,
        feature_class=FeatureClass.BASIC,
        scope=Scope.ONLINE,
    ),
    "gender": FeatureCard(
        name="gender",
        description="Gender of the customer",
        feature_type=FeatureType.CATEGORICAL,
        feature_class=FeatureClass.BASIC,
        scope=Scope.ONLINE,
    ),
    "annual_income": FeatureCard(
        name="annual_income",
        description="Annual income of the customer",
        feature_type=FeatureType.FLOAT64,
        feature_class=FeatureClass.BASIC,
        scope=Scope.ONLINE,
    ),
    "purchase_amount": FeatureCard(
        name="purchase_amount",
        description="Amount of current purchase",
        feature_type=FeatureType.FLOAT64,
        feature_class=FeatureClass.BASIC,
        scope=Scope.ONLINE,
    ),
    "purchase_year": FeatureCard(
        name="purchase_year",
        description="Year of current purchase",
        feature_type=FeatureType.INT32,
        feature_class=FeatureClass.TEMPORAL,
        scope=Scope.ONLINE,
        dependencies=["purchase_date"],
        processor= "date_features"
        
    ),
    "purchase_season": FeatureCard(
        name="purchase_season",
        description="Season of current purchase",
        feature_type=FeatureType.CATEGORICAL,
        feature_class=FeatureClass.TEMPORAL,
        scope=Scope.ONLINE,
        dependencies=["purchase_date"],
        processor= "date_features"
    ),
    "purchase_month": FeatureCard(
        name="purchase_month",
        description="Month of current purchase",
        feature_type=FeatureType.INT32,
        feature_class=FeatureClass.TEMPORAL,
        scope=Scope.ONLINE,
        dependencies=["purchase_date"],
        processor= "date_features"
    ),
    "purchase_day": FeatureCard(
        name="purchase_day",
        description="Day of current purchase",
        feature_type=FeatureType.INT32,
        feature_class=FeatureClass.TEMPORAL,
        scope=Scope.ONLINE,
        dependencies=["purchase_date"],
        processor= "date_features"
    ),
    "purchase_day_of_week": FeatureCard(
        name="purchase_day_of_week",
        description="Day of week of current purchase",
        feature_type=FeatureType.INT32,
        feature_class=FeatureClass.TEMPORAL,
        scope=Scope.ONLINE,
        dependencies=["purchase_date"],
        processor= "date_features"
    ),

    # REALTIME FEATURES (recent history)
    "last_purchase_date": FeatureCard(
        name="last_purchase_date",
        description="Date of last purchase",
        feature_type=FeatureType.DATETIME,
        feature_class=FeatureClass.TEMPORAL,
        scope=Scope.REALTIME,
        dependencies=["purchase_date","customer_id"],
        processor= "last_purchase_date"
        
    ),
    "days_since_last_purchase": FeatureCard(
        name="days_since_last_purchase",
        description="Days since last purchase",
        feature_type=FeatureType.FLOAT64,
        feature_class=FeatureClass.TEMPORAL,
        scope=Scope.REALTIME,
        dependencies=["purchase_date","last_purchase_date"],
        processor= "days_since_last_purchase"
        
    ),
    # TRANSFORMED FEATURES
    "scaled_annual_income": FeatureCard(
        name="scaled_annual_income",
        description="Scaled annual income of the customer",
        feature_type=FeatureType.FLOAT64,
        feature_class=FeatureClass.SCALING,
        scope=Scope.ONLINE,
        dependencies=["annual_income"],
        processor= "scaler"
    ),
    "encoded_gender": FeatureCard(
        name="encoded_gender",
        description="Encoded gender of the customer",
        feature_type=FeatureType.CATEGORICAL,
        feature_class=FeatureClass.ENCODING,
        scope=Scope.ONLINE,
        dependencies=["gender"],
        processor= "categorical_encoder"
    ),
    "encoded_purchase_season": FeatureCard(
        name="encoded_purchase_season",
        description="Encoded season of purchase",
        feature_type=FeatureType.CATEGORICAL,
        feature_class=FeatureClass.ENCODING,
        scope=Scope.ONLINE,
        dependencies=["purchase_season"],
        processor= "categorical_encoder"
    )
}