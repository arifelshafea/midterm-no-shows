"""Gradio interface for the Maple Physio no-show predictor."""

import gradio as gr

try:
    import spaces

    @spaces.GPU
    def zerogpu_placeholder():
        """Placeholder required for Hugging Face ZeroGPU."""
        pass

except ImportError:
    pass

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor


def predict_no_show(
    lead_days,
    age,
    prior_no_shows,
    hour,
    weekday,
    booking,
    insurance,
    treatment,
    first_visit,
    distance_km,
):
    """Validate appointment details and predict no-show risk."""
    try:
        if first_visit not in (0, 1, False, True):
            raise ValueError("first_visit: Must be 0 or 1.")
        appointment = Appointment(
            lead_days=lead_days,
            age=age,
            prior_no_shows=prior_no_shows,
            hour=hour,
            weekday=weekday,
            booking=booking,
            insurance=insurance,
            treatment=treatment,
            first_visit=int(first_visit),
            distance_km=distance_km,
        )

        result = predictor.predict(appointment)

        probability = f"{result.probability:.1%}"
        action = "Call Patient" if result.call_patient else "No Call Needed"

        warning = (
            "Distance was not provided; the model uses the training median."
            if distance_km is None
            else ""
        )

        return probability, action, warning

    except ValueError as error:
        return "", "", f"Invalid appointment: {error}"


# Load the trained model once when the application starts.
predictor = NoShowPredictor()


# Build the user interface.
with gr.Blocks(title="Maple Physio - No-Show Predictor") as demo:
    gr.Markdown("# Maple Physio - No-Show Predictor")
    gr.Markdown("Enter appointment details to predict the risk of a no-show.")

    with gr.Row():
        lead_days = gr.Number(label="Days Before Appointment", value=7, precision=0)
        age = gr.Number(label="Patient Age", value=35, precision=0)

    with gr.Row():
        prior_no_shows = gr.Number(label="Previous No-Shows", value=0, precision=0)

        hour = gr.Dropdown(
            choices=list(range(8, 18)),
            label="Appointment Hour",
            value=9,
        )

    with gr.Row():
        weekday = gr.Dropdown(
            choices=["mon", "tue", "wed", "thu", "fri"],
            label="Appointment Day",
            value="mon",
        )
        booking = gr.Dropdown(
            choices=["online", "phone", "front_desk"],
            label="Booking Method",
            value="online",
        )

    with gr.Row():
        insurance = gr.Dropdown(
            choices=["private", "provincial", "self_pay"],
            label="Insurance Type",
            value="private",
        )
        treatment = gr.Dropdown(
            choices=["physio", "massage", "acupuncture"],
            label="Treatment",
            value="physio",
        )

    with gr.Row():
        first_visit = gr.Checkbox(
            label="First Visit",
            value=False,
        )
        distance_km = gr.Number(
            label="Distance (km) - Optional",
            value=None,
        )

    # Prediction button and results.
    predict_button = gr.Button("Predict No-Show", variant="primary")

    with gr.Row():
        probability_output = gr.Textbox(
            label="No-Show Probability",
            interactive=False,
        )
        action_output = gr.Textbox(
            label="Recommended Action",
            interactive=False,
        )

    warning_output = gr.Textbox(
        label="Warning",
        interactive=False,
    )

    # Connect the button to the prediction function.
    predict_button.click(
        fn=predict_no_show,
        inputs=[
            lead_days,
            age,
            prior_no_shows,
            hour,
            weekday,
            booking,
            insurance,
            treatment,
            first_visit,
            distance_km,
        ],
        outputs=[
            probability_output,
            action_output,
            warning_output,
        ],
    )

if __name__ == "__main__":
    demo.launch()
