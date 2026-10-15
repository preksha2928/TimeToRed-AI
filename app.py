import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from ultralytics import YOLO
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="TimeToRed AI",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 TimeToRed AI")
st.subheader("Predictive Crowd Safety Monitoring System")

st.write(
    "Deep learning-based people detection, crowd density estimation, "
    "heatmap visualization, trend analysis, short-term forecasting, "
    "and predictive crowd safety alerts."
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("yolo11s.pt")


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.info(
        "Check whether yolo11s.pt is present in your project folder."
    )
    st.stop()


# =========================================================
# SIDEBAR CONFIGURATION
# =========================================================
st.sidebar.header("⚙️ Configuration")

zone_name = st.sidebar.text_input(
    "Zone Name",
    "Main Entrance"
)

location_type = st.sidebar.selectbox(
    "Location Type",
    [
        "Entrance",
        "Corridor",
        "Hall",
        "Station",
        "Public Area",
        "Other"
    ]
)


# Fixed YOLO confidence
confidence = 0.10


st.sidebar.subheader("📐 Density Settings")

area_m2 = st.sidebar.number_input(
    "Measured Camera View Area (m²)",
    min_value=0.1,
    value=25.0,
    step=1.0,
    help="Enter the measured real-world area visible to the camera."
)

st.sidebar.caption(
    "Density estimates are meaningful only when the camera-view "
    "area is measured appropriately."
)

# New feature: threshold configuration
st.sidebar.subheader("🚨 Crowd Safety Thresholds")

capacity = st.sidebar.number_input(
    "Configured Crowd Capacity",
    min_value=1,
    value=100,
    step=10,
    help="Set the approved capacity for this monitored zone."
)

warning_percent = st.sidebar.slider(
    "Yellow Warning Threshold (%)",
    min_value=10,
    max_value=100,
    value=60
)

critical_percent = st.sidebar.slider(
    "Red Critical Threshold (%)",
    min_value=10,
    max_value=100,
    value=90
)

if critical_percent < warning_percent:
    st.sidebar.warning(
        "Critical threshold is below warning threshold. "
        "Consider setting it higher."
    )

sound_enabled = st.sidebar.checkbox(
    "Enable Sound Alert Controls",
    value=True
)

st.sidebar.subheader("🎯 Accuracy Evaluation")

evaluate_accuracy = st.sidebar.checkbox(
    "I have a verified reference count"
)

reference_count = None

if evaluate_accuracy:
    reference_count = st.sidebar.number_input(
        "Verified People Count",
        min_value=0,
        value=0,
        step=1
    )


# =========================================================
# PERSON DETECTION
# =========================================================
def detect_people(image_rgb):
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    results = model.predict(
        source=image_bgr,
        conf=confidence,
        classes=[0],
        verbose=False
    )

    annotated = image_rgb.copy()
    centers = []

    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = (
                box.xyxy[0].cpu().numpy().astype(int)
            )

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            centers.append((cx, cy))

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                "Person",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return annotated, len(centers), centers


# =========================================================
# CROWD DENSITY ESTIMATION
# =========================================================
def calculate_density(people_count, area):
    density = people_count / area

    # Prototype categories, not official safety limits.
    if density < 1:
        level = "LOW"
    elif density < 2:
        level = "MODERATE"
    elif density < 4:
        level = "HIGH"
    else:
        level = "VERY HIGH"

    return density, level


# =========================================================
# NEW: PREDICTIVE ALERT CALCULATION
# =========================================================
def calculate_alert(
    current_count,
    predicted_count,
    max_capacity,
    warning_threshold,
    critical_threshold
):
    current_occupancy = (
        current_count / max_capacity
    ) * 100

    predicted_occupancy = (
        predicted_count / max_capacity
    ) * 100

    if current_occupancy >= critical_threshold:
        return (
            "RED",
            "CRITICAL",
            current_occupancy,
            predicted_occupancy
        )

    if predicted_occupancy >= critical_threshold:
        return (
            "ORANGE",
            "PREDICTED CRITICAL",
            current_occupancy,
            predicted_occupancy
        )

    if current_occupancy >= warning_threshold:
        return (
            "YELLOW",
            "WARNING",
            current_occupancy,
            predicted_occupancy
        )

    if predicted_occupancy >= warning_threshold:
        return (
            "YELLOW",
            "EARLY WARNING",
            current_occupancy,
            predicted_occupancy
        )

    return (
        "GREEN",
        "NORMAL",
        current_occupancy,
        predicted_occupancy
    )


# =========================================================
# DETECTION HEATMAP
# =========================================================
def create_heatmap(image_rgb, centers):
    height, width = image_rgb.shape[:2]

    heat = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for x, y in centers:
        cv2.circle(
            heat,
            (x, y),
            45,
            1.0,
            -1
        )

    heat = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=25
    )

    return heat


def display_heatmap(image_rgb, centers):
    heat = create_heatmap(image_rgb, centers)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(image_rgb)
    ax.imshow(
        heat,
        cmap="jet",
        alpha=0.55
    )

    ax.set_title("Crowd Detection Heatmap")
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# ACCURACY EVALUATION
# =========================================================
def display_accuracy(detected, reference):
    error = abs(detected - reference)

    c1, c2, c3 = st.columns(3)

    c1.metric("AI Detected Count", detected)
    c2.metric("Verified Reference", reference)
    c3.metric("Absolute Count Error", error)

    if reference > 0:
        agreement = max(
            0,
            100 * (1 - error / reference)
        )

        st.metric(
            "Count Agreement Estimate",
            f"{agreement:.2f}%"
        )

        st.caption(
            "This is a count-agreement estimate, not a complete "
            "object-detection accuracy metric."
        )

    elif detected == 0:
        st.info("Both counts are zero.")

    else:
        st.info(
            "Percentage error is undefined because "
            "the reference count is zero."
        )


# =========================================================
# NEW: SOUND ALERT CONTROLS
# =========================================================


def show_sound_alert(alert_level):
    if not sound_enabled:
        st.info("Sound alerts are disabled in settings.")
        return

    if alert_level == "GREEN":
        st.success("Normal status: no alarm required.")
        return

    if alert_level == "YELLOW":
        message = "Warning alert: 5-second alarm"
        frequency = 660

    elif alert_level == "ORANGE":
        message = "High alert: 5-second alarm"
        frequency = 780

    else:
        message = "Critical alert: 5-second alarm"
        frequency = 950

    st.warning(message)

    components.html(
        f"""
        <div style="font-family:Arial,sans-serif">
          <b>🔊 {message}</b>
          <p id="soundStatus">Playing alarm...</p>
        </div>

        <script>
        (async function() {{
          let context = null;

          try {{
            context = new (
              window.AudioContext || window.webkitAudioContext
            )();

            await context.resume();

            const oscillator = context.createOscillator();
            const gain = context.createGain();

            oscillator.type = "sine";
            oscillator.frequency.value = {frequency};

            gain.gain.setValueAtTime(
              0.15, context.currentTime
            );

            oscillator.connect(gain);
            gain.connect(context.destination);

            oscillator.start();
            oscillator.stop(context.currentTime + 5);

            oscillator.onended = () => {{
              document.getElementById("soundStatus").textContent =
                "5-second alarm completed.";

              context.close();
            }};

          }} catch (error) {{
            document.getElementById("soundStatus").textContent =
              "Browser blocked automatic audio playback.";

            if (context) {{
              context.close();
            }}
          }}
        }})();
        </script>
        """,
        height=90
    )


# =========================================================
# MODE SELECTION
# =========================================================
mode = st.radio(
    "Select Analysis Mode",
    ["Image Analysis", "Video Analysis"],
    horizontal=True
)


# =========================================================
# IMAGE ANALYSIS
# =========================================================
if mode == "Image Analysis":

    st.header("🖼️ Image Crowd Analysis")

    uploaded_image = st.file_uploader(
        "Upload a crowd image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        image_rgb = np.array(image)

        with st.spinner("Detecting people..."):
            annotated, count, centers = detect_people(
                image_rgb
            )

        st.success("Image analysis completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                image_rgb,
                use_container_width=True
            )

        with col2:
            st.subheader("AI Detection")
            st.image(
                annotated,
                use_container_width=True
            )

        density, level = calculate_density(
            count,
            area_m2
        )

        # A single image cannot estimate movement over 10 seconds.
        # Keep predicted count equal to current count as a neutral baseline.
        estimated_count = count
        estimated_density = estimated_count / area_m2

        (
            alert_color,
            alert_status,
            occupancy,
            predicted_occupancy
        ) = calculate_alert(
            count,
            estimated_count,
            capacity,
            warning_percent,
            critical_percent
        )

        st.divider()
        st.subheader("📊 Crowd Analysis Dashboard")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Actual Count (AI Detected)",
            count
        )

        c2.metric(
            "Crowd Density",
            f"{density:.2f} people/m²"
        )

        c3.metric(
            "Density Level",
            level
        )
        st.metric(
            "Remaining Crowd Capacity",
            f"{max(0, capacity - count)} people"
        )

        st.metric(
            "Remaining Capacity",
            f"{max(0, 100 - occupancy):.1f}%"
        )

        st.subheader("🚨 Predictive Safety Status")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Estimated Count (+10 sec)",
            estimated_count
        )

        c2.metric(
            "Estimated Density",
            f"{estimated_density:.2f} people/m²"
        )

        c3.metric(
            "Configured Threshold",
            f"{capacity} people"
        )

        st.metric(
            "Current Occupancy",
            f"{occupancy:.1f}%"
        )

        if alert_color == "RED":
            st.error(f"🔴 {alert_status}")
        elif alert_color == "ORANGE":
            st.warning(f"🟠 {alert_status}")
        elif alert_color == "YELLOW":
            st.warning(f"🟡 {alert_status}")
        else:
            st.success("🟢 NORMAL")

        st.caption(
            "A still image cannot establish crowd movement. "
            "Its +10-second count is a baseline, not a movement forecast."
        )

        show_sound_alert(alert_color)

        st.subheader("🔥 Crowd Heatmap")
        display_heatmap(
            image_rgb,
            centers
        )

        st.subheader("📋 Detection Summary")

        summary = pd.DataFrame({
            "Metric": [
                "Zone",
                "Location Type",
                "AI Detected People",
                "Measured Area (m²)",
                "Estimated Density (people/m²)",
                "Density Category",
                "Configured Capacity",
                "Current Occupancy (%)",
                "Threshold Status",
                "Alert",
                "Evaluation Status"
            ],
            "Value": [
                zone_name,
                location_type,
                count,
                area_m2,
                round(density, 2),
                level,
                capacity,
                round(occupancy, 1),
                alert_status,
                alert_color,
                (
                    "Reference count available"
                    if evaluate_accuracy
                    else "Not independently verified"
                )
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True
        )

        if evaluate_accuracy and reference_count is not None:
            st.subheader("🎯 Count Evaluation")
            display_accuracy(
                count,
                int(reference_count)
            )

        image_csv = summary.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Image Analysis CSV",
            data=image_csv,
            file_name="timetored_image_analysis.csv",
            mime="text/csv"
        )


# =========================================================
# VIDEO ANALYSIS
# =========================================================
else:

    st.header("🎥 Video Crowd Analysis")

    uploaded_video = st.file_uploader(
        "Upload a crowd video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    # Fixed sampling interval: one frame every second
    sample_interval = 1

    if uploaded_video is not None:

        start_analysis = st.button(
            "▶️ Start Crowd Analysis",
            type="primary"
        )

        if start_analysis:

            temp_path = None
            cap = None
            records = []
            latest_rgb = None
            latest_centers = []

            try:
                suffix = os.path.splitext(
                    uploaded_video.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:
                    temp_file.write(
                        uploaded_video.getvalue()
                    )
                    temp_path = temp_file.name

                cap = cv2.VideoCapture(temp_path)

                if not cap.isOpened():
                    st.error(
                        "Unable to open the uploaded video."
                    )
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )

                if not np.isfinite(fps) or fps <= 0:
                    fps = 25.0

                frame_interval = max(
                1,
                int(round(fps * 1.0))
                )

                total_samples = max(
                    1,
                    int(total_frames / frame_interval)
                )

                progress = st.progress(0)
                status_text = st.empty()
                preview = st.empty()

                frame_index = 0
                sample_number = 0

                while cap.isOpened():

                    success, frame = cap.read()

                    if not success:
                        break

                    if frame_index % frame_interval == 0:

                        frame_rgb = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                        annotated, count, centers = detect_people(
                            frame_rgb
                        )

                        timestamp = frame_index / fps

                        density, level = calculate_density(
                            count,
                            area_m2
                        )

                        latest_rgb = frame_rgb
                        latest_centers = centers

                        records.append({
                            "Time (seconds)": round(
                                timestamp, 2
                            ),
                            "Actual Count (AI Detected)": count,
                            "Density (people/m²)": round(
                                density, 3
                            ),
                            "Density Level": level
                        })

                        preview.image(
                            annotated,
                            caption=(
                                f"Time: {timestamp:.1f}s | "
                                f"People: {count} | "
                                f"Density: {density:.2f} people/m²"
                            ),
                            use_container_width=True
                        )

                        sample_number += 1

                        progress.progress(
                            min(
                                1.0,
                                sample_number / total_samples
                            )
                        )

                        status_text.write(
                            f"Analyzed samples: {sample_number}"
                        )

                    frame_index += 1

                cap.release()
                cap = None

                if not records:
                    st.error(
                        "No video frames were analyzed."
                    )
                    st.stop()

                df = pd.DataFrame(records)

                counts = df[
                    "Actual Count (AI Detected)"
                ].to_numpy(dtype=float)

                times = df[
                    "Time (seconds)"
                ].to_numpy(dtype=float)

                current_count = int(counts[-1])

                current_density, current_level = calculate_density(
                    current_count,
                    area_m2
                )

                # ---------------------------------------------
                # TREND AND FORECAST
                # ---------------------------------------------
                if (
                    len(counts) >= 2
                    and len(np.unique(times)) >= 2
                ):

                    recent_size = min(
                        5,
                        len(counts)
                    )

                    recent_counts = counts[-recent_size:]
                    recent_times = times[-recent_size:]

                    slope = np.polyfit(
                        recent_times,
                        recent_counts,
                        1
                    )[0]

                    previous_count = int(counts[-2])
                    change = current_count - previous_count

                    # Existing 10-second linear trend estimate
                    estimated_count = max(
                        0,
                        int(round(current_count + slope * 10))
                    )

                else:
                    slope = 0.0
                    previous_count = current_count
                    change = 0
                    estimated_count = current_count

                trend = (
                    "INCREASING 📈"
                    if slope > 0.25
                    else "DECREASING 📉"
                    if slope < -0.25
                    else "STABLE ➡️"
                )

                rapid_increase = (
                    change >= 3
                    or (
                        previous_count > 0
                        and change / previous_count >= 0.20
                    )
                )

                if rapid_increase:
                    crowd_status = "WARNING — RAPID INCREASE"
                elif trend.startswith("INCREASING"):
                    crowd_status = "INCREASING — MONITOR"
                elif trend.startswith("DECREASING"):
                    crowd_status = "DECREASING"
                else:
                    crowd_status = "STABLE"

                estimated_density = (
                    estimated_count / area_m2
                )

                (
                    alert_color,
                    alert_status,
                    occupancy,
                    predicted_occupancy
                ) = calculate_alert(
                    current_count,
                    estimated_count,
                    capacity,
                    warning_percent,
                    critical_percent
                )

                # ---------------------------------------------
                # DASHBOARD
                # ---------------------------------------------
                st.divider()
                st.subheader("📊 Crowd Monitoring Dashboard")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Actual Count (AI Detected)",
                    current_count,
                    delta=change
                )

                c2.metric(
                    "Estimated Count (+10 sec)",
                    estimated_count,
                    delta=estimated_count - current_count
                )

                c3.metric(
                    "Crowd Density",
                    f"{current_density:.2f} people/m²"
                )

                c4, c5 = st.columns(2)

                c4.metric(
                    "Density Level",
                    current_level
                )

                c5.metric(
                    "Crowd Trend",
                    trend
                )

                if rapid_increase:
                    st.warning(f"⚠️ {crowd_status}")
                elif trend.startswith("INCREASING"):
                    st.info(f"👀 {crowd_status}")
                elif trend.startswith("DECREASING"):
                    st.success(f"📉 {crowd_status}")
                else:
                    st.success(f"✅ {crowd_status}")

                st.caption(
                    "Density categories and rapid-increase alerts are "
                    "prototype heuristics, not certified safety thresholds."
                )

                # ---------------------------------------------
                # NEW PREDICTIVE SAFETY DASHBOARD
                # ---------------------------------------------
                st.subheader("🚨 Predictive Safety Status")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Configured Capacity",
                    capacity
                )

                c2.metric(
                    "Current Occupancy",
                    f"{occupancy:.1f}%"
                )

                c3.metric(
                    "Predicted Occupancy",
                    f"{predicted_occupancy:.1f}%"
                )

                c4.metric(
                    "Estimated Density (+10s)",
                    f"{estimated_density:.2f} people/m²"
                )

                if alert_color == "RED":
                    st.error(f"🔴 CRITICAL ALERT: {alert_status}")
                elif alert_color == "ORANGE":
                    st.warning(f"🟠 HIGH ALERT: {alert_status}")
                elif alert_color == "YELLOW":
                    st.warning(f"🟡 WARNING: {alert_status}")
                else:
                    st.success("🟢 NORMAL: Crowd within configured thresholds")

                show_sound_alert(alert_color)

                # ---------------------------------------------
                # COUNT GRAPH
                # ---------------------------------------------
                st.subheader("📈 Crowd Count Trend")

                fig, ax = plt.subplots(figsize=(10, 5))

                ax.plot(
                    times,
                    counts,
                    marker="o",
                    label="AI Detected Count"
                )

                if len(counts) >= 2:
                    trend_line = np.polyval(
                        np.polyfit(times, counts, 1),
                        times
                    )

                    ax.plot(
                        times,
                        trend_line,
                        linestyle="--",
                        label="Fitted Trend"
                    )

                ax.set_xlabel("Video Time (seconds)")
                ax.set_ylabel("Detected People")
                ax.set_title("People Count Over Time")
                ax.grid(True, alpha=0.3)
                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

                # ---------------------------------------------
                # DENSITY GRAPH
                # ---------------------------------------------
                st.subheader("📐 Crowd Density Trend")

                fig, ax = plt.subplots(figsize=(10, 4))

                ax.plot(
                    times,
                    df["Density (people/m²)"],
                    marker="o",
                    label="Estimated Density"
                )

                ax.set_xlabel("Video Time (seconds)")
                ax.set_ylabel("People per m²")
                ax.set_title("Density Over Time")
                ax.grid(True, alpha=0.3)
                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

                # ---------------------------------------------
                # HEATMAP
                # ---------------------------------------------
                st.subheader("🔥 Latest Analyzed Frame Heatmap")

                if latest_rgb is not None:
                    display_heatmap(
                        latest_rgb,
                        latest_centers
                    )

                # ---------------------------------------------
                # RESULTS TABLE
                # ---------------------------------------------
                st.subheader("📋 Frame-by-Frame Results")

                df["Change vs Previous"] = (
                    df["Actual Count (AI Detected)"]
                    .diff()
                    .fillna(0)
                    .astype(int)
                )

                df["Trend Estimate"] = np.where(
                    df["Change vs Previous"] > 0,
                    "Increasing",
                    np.where(
                        df["Change vs Previous"] < 0,
                        "Decreasing",
                        "Stable"
                    )
                )

                # New columns for each sampled frame
                row_counts = df[
                    "Actual Count (AI Detected)"
                ].to_numpy(dtype=float)

                row_times = df[
                    "Time (seconds)"
                ].to_numpy(dtype=float)

                row_predictions = []

                for i in range(len(row_counts)):
                    start = max(0, i - 4)

                    recent_counts = row_counts[start:i + 1]
                    recent_times = row_times[start:i + 1]

                    if (
                        len(recent_counts) >= 2
                        and len(np.unique(recent_times)) >= 2
                    ):
                        row_slope = np.polyfit(
                            recent_times,
                            recent_counts,
                            1
                        )[0]

                        prediction = max(
                            0,
                            int(round(
                                row_counts[i] + row_slope * 10
                            ))
                        )
                    else:
                        prediction = int(row_counts[i])

                    row_predictions.append(prediction)

                df["Estimated People Count (+10s)"] = (
                    row_predictions
                )

                df["Estimated Density (people/m²)"] = (
                    df["Estimated People Count (+10s)"] / area_m2
                ).round(3)

                df["Threshold"] = capacity

                df["Threshold Status"] = (
                    df["Actual Count (AI Detected)"]
                    / capacity * 100
                ).round(1).astype(str) + "% occupied"

                def row_alert(row):
                    actual = row[
                        "Actual Count (AI Detected)"
                    ]

                    predicted = row[
                        "Estimated People Count (+10s)"
                    ]

                    if actual >= capacity * critical_percent / 100:
                        return "RED - CRITICAL"

                    if predicted >= capacity * critical_percent / 100:
                        return "ORANGE - PREDICTED CRITICAL"

                    if actual >= capacity * warning_percent / 100:
                        return "YELLOW - WARNING"

                    if predicted >= capacity * warning_percent / 100:
                        return "YELLOW - EARLY WARNING"

                    return "GREEN - NORMAL"

                df["Alert"] = df.apply(
                    row_alert,
                    axis=1
                )

                df["Evaluation Status"] = (
                    "Not independently verified"
                )

                if evaluate_accuracy and reference_count is not None:
                    df["Evaluation Status"] = (
                        "Reference count available; "
                        "latest frame comparison only"
                    )

                # Display requested columns in the results table
                display_columns = [
                    "Time (seconds)",
                    "Actual Count (AI Detected)",
                    "Estimated People Count (+10s)",
                    "Density (people/m²)",
                    "Estimated Density (people/m²)",
                    "Density Level",
                    "Threshold",
                    "Threshold Status",
                    "Alert",
                    "Trend Estimate",
                    "Evaluation Status",
                    "Change vs Previous"
                ]

                st.dataframe(
                    df[display_columns],
                    use_container_width=True
                )

                # ---------------------------------------------
                # OPTIONAL ACCURACY EVALUATION
                # ---------------------------------------------
                if evaluate_accuracy and reference_count is not None:
                    st.subheader("🎯 Count Evaluation")

                    display_accuracy(
                        current_count,
                        int(reference_count)
                    )

                # ---------------------------------------------
                # DOWNLOAD RESULTS
                # ---------------------------------------------
                csv_data = df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Crowd Analysis CSV",
                    data=csv_data,
                    file_name="timetored_crowd_analysis.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Video analysis error: {e}")

            finally:
                if cap is not None:
                    cap.release()

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "TimeToRed AI | Crowd-monitoring prototype. "
    "Results depend on camera angle, visibility, model performance "
    "and correct area measurement. Validate before real-world use."
)