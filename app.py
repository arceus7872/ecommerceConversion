from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import shap
import os


app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "ecommerce_conversion_model.pkl"
)

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# SHAP EXPLAINER
# ============================================================

rf_model = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

explainer = shap.TreeExplainer(rf_model)

print("SHAP explainer created successfully!")


# ============================================================
# FEATURES
# ============================================================

COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]


# ============================================================
# FEATURE NAME CLEANING
# ============================================================

def clean_feature_name(feature_name):

    name = feature_name

    name = name.replace("num__", "")
    name = name.replace("cat__", "")
    name = name.replace("_", " ")

    replacements = {
        "PageValues": "Page Value",
        "ProductRelated": "Product Related",
        "ProductRelated Duration": "Product Related Duration",
        "Administrative Duration": "Administrative Duration",
        "Informational Duration": "Informational Duration",
        "BounceRates": "Bounce Rate",
        "ExitRates": "Exit Rate",
        "VisitorType": "Visitor Type",
        "OperatingSystems": "Operating System",
        "TrafficType": "Traffic Type"
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return name.strip()


# ============================================================
# SHAP FACTORS
# ============================================================

def get_shap_factors(input_df):

    transformed = preprocessor.transform(input_df)

    if hasattr(transformed, "toarray"):
        transformed_for_shap = transformed.toarray()
    else:
        transformed_for_shap = np.asarray(transformed)

    transformed_for_shap = transformed_for_shap.astype(float)

    shap_values = explainer.shap_values(
        transformed_for_shap,
        check_additivity=False
    )

    # --------------------------------------------------------
    # GET SHAP VALUES FOR PURCHASE CLASS
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        values = shap_values[1][0]

    elif isinstance(shap_values, np.ndarray):

        if shap_values.ndim == 3:

            if shap_values.shape[2] >= 2:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0, :, 0]

        elif shap_values.ndim == 2:

            values = shap_values[0]

        else:

            values = shap_values.flatten()

    else:

        values = np.asarray(shap_values).flatten()


    feature_names = preprocessor.get_feature_names_out()


    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

    numerical_features = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay"
    ]


    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    categorical_features = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend"
    ]


    factors = []


    # --------------------------------------------------------
    # NUMERICAL SHAP FEATURES
    # --------------------------------------------------------

    for feature_name, shap_value in zip(
        feature_names,
        values
    ):

        original_name = feature_name.replace(
            "num__",
            ""
        )

        original_name = original_name.replace(
            "cat__",
            ""
        )

        if original_name in numerical_features:

            factors.append({
                "name": clean_feature_name(
                    feature_name
                ),
                "value": float(shap_value)
            })


    # --------------------------------------------------------
    # CATEGORICAL SHAP FEATURES
    # --------------------------------------------------------

    for categorical_feature in categorical_features:

        category_shap_total = 0.0

        found = False

        for feature_name, shap_value in zip(
            feature_names,
            values
        ):

            cleaned_name = feature_name.replace(
                "cat__",
                ""
            )

            if cleaned_name.startswith(
                categorical_feature + "_"
            ):

                category_shap_total += float(
                    shap_value
                )

                found = True


        if found:

            selected_value = input_df.iloc[0][
                categorical_feature
            ]


            # ------------------------------------------------
            # FORMAT BOOLEAN VALUE
            # ------------------------------------------------

            if categorical_feature == "Weekend":

                selected_value = (
                    "Yes"
                    if bool(selected_value)
                    else "No"
                )


            # ------------------------------------------------
            # ADD GROUPED CATEGORICAL FEATURE
            # ------------------------------------------------

            factors.append({

                "name":
                    f"{clean_feature_name(categorical_feature)}: "
                    f"{selected_value}",

                "value":
                    category_shap_total

            })


    # --------------------------------------------------------
    # SORT BY ABSOLUTE SHAP IMPACT
    # --------------------------------------------------------

    factors.sort(
        key=lambda x: abs(x["value"]),
        reverse=True
    )


    # --------------------------------------------------------
    # POSITIVE FACTORS
    # --------------------------------------------------------

    positive_factors = [
        factor
        for factor in factors
        if factor["value"] > 0
    ][:3]


    # --------------------------------------------------------
    # NEGATIVE FACTORS
    # --------------------------------------------------------

    negative_factors = [
        factor
        for factor in factors
        if factor["value"] < 0
    ][:3]


    # --------------------------------------------------------
    # FORMAT SHAP VALUES
    # --------------------------------------------------------

    for factor in positive_factors:

        factor["value"] = (
            f"{factor['value']:.4f}"
        )


    for factor in negative_factors:

        factor["value"] = (
            f"{factor['value']:.4f}"
        )


    return positive_factors, negative_factors


# ============================================================
# SESSION ANALYSIS
# ============================================================

def analyze_session(data):

    product_pages = data["ProductRelated"]
    product_duration = data["ProductRelated_Duration"]

    total_pages = (
        data["Administrative"]
        + data["Informational"]
        + data["ProductRelated"]
    )

    total_duration = (
        data["Administrative_Duration"]
        + data["Informational_Duration"]
        + data["ProductRelated_Duration"]
    )

    bounce_rate = data["BounceRates"]
    exit_rate = data["ExitRates"]


    # --------------------------------------------------------
    # PRODUCT INTEREST
    # --------------------------------------------------------

    if product_pages >= 29 or product_duration >= 1075.41:

        product_interest = "High"

    elif product_pages >= 10 or product_duration >= 298:

        product_interest = "Medium"

    else:

        product_interest = "Low"


    # --------------------------------------------------------
    # ENGAGEMENT
    # --------------------------------------------------------

    if total_pages >= 32 or total_duration >= 1199.72:

        engagement = "High"

    elif total_pages >= 12 or total_duration >= 354.08:

        engagement = "Medium"

    else:

        engagement = "Low"


    # --------------------------------------------------------
    # EXIT RISK
    # --------------------------------------------------------

    if exit_rate > 0.036923 or bounce_rate > 0.01:

        exit_risk = "High"

    elif exit_rate > 0.017244 or bounce_rate > 0:

        exit_risk = "Medium"

    else:

        exit_risk = "Low"


    return (
        product_interest,
        engagement,
        exit_risk,
        total_pages,
        total_duration
    )


# ============================================================
# SESSION SIGNALS
# ============================================================

def generate_signals(
    data,
    product_interest,
    engagement,
    exit_risk
):

    positive_signals = []
    risk_signals = []


    if product_interest == "High":

        positive_signals.append(
            "Strong product browsing activity."
        )

    elif product_interest == "Medium":

        positive_signals.append(
            "Moderate product browsing activity."
        )

    else:

        risk_signals.append(
            "Limited product browsing activity."
        )


    if engagement == "High":

        positive_signals.append(
            "High overall session engagement."
        )

    elif engagement == "Medium":

        positive_signals.append(
            "Moderate overall session engagement."
        )

    else:

        risk_signals.append(
            "Low overall session engagement."
        )


    if exit_risk == "Low":

        positive_signals.append(
            "Low bounce and exit-rate signals."
        )

    elif exit_risk == "Medium":

        risk_signals.append(
            "Moderate exit-rate or bounce-rate risk."
        )

    else:

        risk_signals.append(
            "High exit or bounce-rate risk."
        )


    if data["PageValues"] > 0:

        positive_signals.append(
            "The session has measurable page value."
        )

    else:

        risk_signals.append(
            "No page value is associated with this session."
        )


    return positive_signals, risk_signals


# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    probability_percentage,
    data,
    product_interest,
    engagement,
    exit_risk
):

    recommendations = []


    if probability_percentage >= 60:

        recommendations.append(
            "Prioritize this visitor with personalized product recommendations."
        )


        if data["PageValues"] > 0:

            recommendations.append(
                "Consider showing a relevant offer or checkout incentive."
            )


        if product_interest == "High":

            recommendations.append(
                "Highlight complementary or closely related products."
            )


        if exit_risk == "High":

            recommendations.append(
                "Reduce checkout or navigation friction because exit risk is elevated."
            )


    elif probability_percentage >= 30:

        recommendations.append(
            "Use personalized product recommendations to increase engagement."
        )


        if product_interest in ["Medium", "High"]:

            recommendations.append(
                "Show products related to the visitor's current browsing behavior."
            )


        if exit_risk in ["Medium", "High"]:

            recommendations.append(
                "Improve calls-to-action and reduce potential page friction."
            )


        if data["PageValues"] == 0:

            recommendations.append(
                "Consider targeted content or offers to increase conversion value."
            )


    else:

        recommendations.append(
            "Focus on improving engagement before using aggressive conversion offers."
        )


        if product_interest == "Low":

            recommendations.append(
                "Use relevant product recommendations to increase product discovery."
            )


        if engagement == "Low":

            recommendations.append(
                "Improve navigation and calls-to-action to encourage deeper browsing."
            )


        if exit_risk in ["Medium", "High"]:

            recommendations.append(
                "Investigate pages where visitors are leaving and reduce potential friction."
            )


    return recommendations[:4]


# ============================================================
# CREATE DATA FROM FORM
# ============================================================

def get_data_from_form(form):

    data = {

        "Administrative":
            float(form["Administrative"]),

        "Administrative_Duration":
            float(form["Administrative_Duration"]),

        "Informational":
            float(form["Informational"]),

        "Informational_Duration":
            float(form["Informational_Duration"]),

        "ProductRelated":
            float(form["ProductRelated"]),

        "ProductRelated_Duration":
            float(form["ProductRelated_Duration"]),

        "BounceRates":
            float(form["BounceRates"]),

        "ExitRates":
            float(form["ExitRates"]),

        "PageValues":
            float(form["PageValues"]),

        "SpecialDay":
            float(form["SpecialDay"]),

        "Month":
            form["Month"],

        "OperatingSystems":
            int(form["OperatingSystems"]),

        "Browser":
            int(form["Browser"]),

        "Region":
            int(form["Region"]),

        "TrafficType":
            int(form["TrafficType"]),

        "VisitorType":
            form["VisitorType"],

        "Weekend":
            form["Weekend"] == "True"
    }

    return data


# ============================================================
# RUN COMPLETE ANALYSIS
# ============================================================

def analyze_prediction(data):

    input_df = pd.DataFrame(
        [data],
        columns=COLUMNS
    )


    # --------------------------------------------------------
    # PREDICT PROBABILITY
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_df
    )[0][1]


    probability_percentage = round(
        float(probability) * 100,
        2
    )


    # --------------------------------------------------------
    # INTENT LEVEL
    # --------------------------------------------------------

    if probability_percentage >= 60:

        intent_level = "High Purchase Intent"

    elif probability_percentage >= 30:

        intent_level = "Medium Purchase Intent"

    else:

        intent_level = "Low Purchase Intent"


    # --------------------------------------------------------
    # SESSION ANALYSIS
    # --------------------------------------------------------

    (
        product_interest,
        engagement,
        exit_risk,
        total_pages,
        total_duration
    ) = analyze_session(data)


    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    positive_signals, risk_signals = generate_signals(
        data,
        product_interest,
        engagement,
        exit_risk
    )


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    positive_factors, negative_factors = get_shap_factors(
        input_df
    )


    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = generate_recommendations(
        probability_percentage,
        data,
        product_interest,
        engagement,
        exit_risk
    )


    # --------------------------------------------------------
    # EXPLANATION SUMMARY
    # --------------------------------------------------------

    if probability_percentage >= 60:

        explanation_summary = (
            "This visitor shows strong purchase intent based on "
            "the model's estimated probability and observed "
            "session behavior."
        )

    elif probability_percentage >= 30:

        explanation_summary = (
            "This visitor shows moderate purchase intent. "
            "The session contains some positive conversion "
            "signals but also areas that could be improved."
        )

    else:

        explanation_summary = (
            "This visitor currently shows relatively low "
            "purchase intent. Improving engagement and "
            "reducing friction may increase the likelihood "
            "of conversion."
        )


    return {

        "result":
            intent_level,

        "probability":
            probability_percentage,

        "product_interest":
            product_interest,

        "engagement":
            engagement,

        "exit_risk":
            exit_risk,

        "positive_signals":
            positive_signals,

        "risk_signals":
            risk_signals,

        "explanation_summary":
            explanation_summary,

        "positive_factors":
            positive_factors,

        "negative_factors":
            negative_factors,

        "recommendations":
            recommendations,

        "total_pages":
            total_pages,

        "total_duration":
            round(
                total_duration,
                2
            ),

        "page_value":
            data["PageValues"],

        # Keep original session for simulator
        "session_data":
            data
    }


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = get_data_from_form(
            request.form
        )

        analysis = analyze_prediction(
            data
        )

        return render_template(
            "result.html",
            **analysis
        )


    except Exception as e:

        return f"""
        <h2>Error while making prediction</h2>
        <p>{str(e)}</p>
        <a href="/">Go Back</a>
        """, 400


# ============================================================
# WHAT-IF SIMULATOR
# ============================================================

@app.route(
    "/simulate",
    methods=["POST"]
)
def simulate():

    try:

        # ----------------------------------------------------
        # Recover original visitor data
        # ----------------------------------------------------

        data = {}


        for column in COLUMNS:

            value = request.form.get(
                f"original_{column}"
            )


            if value is None:

                raise ValueError(
                    f"Missing original value for {column}"
                )


            if column in [
                "Month",
                "VisitorType"
            ]:

                data[column] = value


            elif column == "Weekend":

                data[column] = (
                    value == "True"
                )


            elif column in [
                "OperatingSystems",
                "Browser",
                "Region",
                "TrafficType"
            ]:

                data[column] = int(
                    float(value)
                )


            else:

                data[column] = float(
                    value
                )


        # ----------------------------------------------------
        # What-if features
        # ----------------------------------------------------

        simulation_features = [

            "ProductRelated",

            "ProductRelated_Duration",

            "BounceRates",

            "ExitRates",

            "PageValues"

        ]


        for feature in simulation_features:

            what_if_value = request.form.get(
                f"whatif_{feature}"
            )


            if what_if_value not in [
                None,
                ""
            ]:

                data[feature] = float(
                    what_if_value
                )


        # ----------------------------------------------------
        # ORIGINAL DATA
        # ----------------------------------------------------

        original_data = data.copy()


        for feature in simulation_features:

            original_value = request.form.get(
                f"original_{feature}"
            )


            if original_value not in [
                None,
                ""
            ]:

                original_data[feature] = float(
                    original_value
                )


        original_df = pd.DataFrame(
            [original_data],
            columns=COLUMNS
        )


        # ----------------------------------------------------
        # ORIGINAL PROBABILITY
        # ----------------------------------------------------

        original_probability = (

            model.predict_proba(
                original_df
            )[0][1]

            * 100

        )


        # ----------------------------------------------------
        # SIMULATED DATA
        # ----------------------------------------------------

        simulated_df = pd.DataFrame(
            [data],
            columns=COLUMNS
        )


        # ----------------------------------------------------
        # SIMULATED PROBABILITY
        # ----------------------------------------------------

        simulated_probability = (

            model.predict_proba(
                simulated_df
            )[0][1]

            * 100

        )


        # ----------------------------------------------------
        # ROUND VALUES
        # ----------------------------------------------------

        original_probability = round(
            float(original_probability),
            2
        )


        simulated_probability = round(
            float(simulated_probability),
            2
        )


        probability_change = round(
            simulated_probability
            - original_probability,
            2
        )


        # ----------------------------------------------------
        # CHANGE DIRECTION
        # ----------------------------------------------------

        if probability_change > 0:

            change_direction = "increase"

        elif probability_change < 0:

            change_direction = "decrease"

        else:

            change_direction = "no_change"


        # ----------------------------------------------------
        # RE-RUN COMPLETE ANALYSIS
        # ----------------------------------------------------

        analysis = analyze_prediction(
            data
        )


        # ----------------------------------------------------
        # RESULT PAGE
        # ----------------------------------------------------

        return render_template(

            "result.html",

            **analysis,

            simulation_run=True,

            original_probability=
                original_probability,

            simulated_probability=
                simulated_probability,

            probability_change=
                probability_change,

            change_direction=
                change_direction,

            simulation_data=
                data,

            original_data=
                original_data

        )


    except Exception as e:

        return f"""
        <h2>Error while running simulation</h2>
        <p>{str(e)}</p>
        <a href="/">Go Back</a>
        """, 400


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.route(
    "/model-performance",
    methods=["GET"]
)
def model_performance():

    return render_template(
        "model.html"
    )


# ============================================================
# ABOUT
# ============================================================

@app.route(
    "/about",
    methods=["GET"]
)
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
