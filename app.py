import streamlit as st
from pathlib import Path
import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parent

MODEL = ROOT / "artifacts" / "models" / "model.joblib"

DATA_PATH = ROOT / "data" / "BTSsilom_condo.xlsx"

UNIT_ROOM_SIZE_RANGE = {
    "Studio": (20.0, 127.0),
    "1 Bedroom": (20.0, 400.0),
    "Duplex 1 Bedroom": (30.0, 366.0),
    "2 Bedroom": (30.0, 277.0),
    "Duplex 2 Bedroom": (42.0, 168.0),
    "3 Bedroom": (52.0, 453.33),
    "Duplex 3 Bedroom": (110.0, 120.0),
    "4 Bedroom": (120.0, 440.0),
    "Duplex 4 Bedroom": (366.0, 385.0),
    "Moff": (78.0, 78.0),
    "Penthouse": (230.0, 354.0),
}

def room_size_group(size: float) -> str:
    if size <= 25:
        return "XS"
    if size <= 35:
        return "S"
    if size <= 50:
        return "M"
    if size <= 80:
        return "L"
    if size <= 150:
        return "XL"
    else:
        return "XXL"
    

@st.cache_resource
def load_model():
    if MODEL.exists():
        return joblib.load(MODEL), MODEL

    raise FileNotFoundError(
        f"Model file not found: {MODEL}\n"
    )
    
@st.cache_data
def load_project_options():
    if not DATA_PATH.exists():
        return ["Other"]

    df = pd.read_excel(DATA_PATH, usecols=["project_name"])

    projects = (
        df["project_name"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
    )

    counts = projects.value_counts()

    popular_projects = counts[counts >= 100].index.astype(str).tolist()

    return ["Other"] + sorted(popular_projects)


def get_category_options(model):
    fallback = {
        "poster_status": ["Verified", "Non Verified", "No Registered"],
        "unit_type": ['Studio'],
        "project_name_grouped": ["Other"],
        "room_size_group": ["XS", "S", "M", "L", "XL", "XXL"]
    }

    try:
        preprocessor = model.named_steps['preprocessor']

        for name, transformer, columns in preprocessor.transformers_:
            if name == "cat":
                options = {}
                for col, values in zip(columns, transformer.categories_):
                    options[col] = [str(v) for v in values]
                
                return options
    except Exception:
        pass


    return fallback

@st.cache_data
def load_unit_type_options():
    if not DATA_PATH.exists():
        return ["Condo"]

    df = pd.read_excel(DATA_PATH, usecols=["unit_type"])

    unit_types = (
        df["unit_type"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return sorted(unit_types)

def select_with_default(label, options, default):
    options = list(options)
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index)


st.set_page_config(
    page_title = "Condo Rental Prediction Along BTS Silom",
    page_icon = "🏙️",
    layout='wide'
)

st.title("🏙️ Condo Rental Prediction Along BTS Silom")
st.write("Estimate monthly rental price from condo features.")

try:
    model, model_path = load_model()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

category_options = get_category_options(model)
popular_projects = load_project_options()

st.caption(f"Loaded Model: '{model_path.relative_to(ROOT)}'")

with st.sidebar:
    st.header("Condo details")

    # no_bedroom = st.number_input(
    #     "Number of bedrooms",
    #     min_value=0.0,
    #     max_value=4.0,
    #     value=1.0,
    #     step=1.0
    # )

    # floor = st.number_input(
    #     "Floor",
    #     min_value=1.0,
    #     max_value=30.0,
    #     value=10.0,
    #     step=1.0
    # )

    unit_type = st.selectbox(
        "Unit type",
        options=list(UNIT_ROOM_SIZE_RANGE.keys())
    )

    room_min, room_max = UNIT_ROOM_SIZE_RANGE[unit_type]

    st.caption(f"Allowed room size for {unit_type}: {room_min:g} - {room_max:g}")

    if room_min == room_max:
        room_size = room_min
        st.number_input(
            "Room size",
            value=float(room_size),
            disabled=True,
        )

    else:
        default_room_size = (room_min + room_max) / 2

        room_size = st.slider(
            "Room size",
            min_value=float(room_min),
            max_value=float(room_max),
            value=float(default_room_size),
            step=0.5
        )

    poster_status = select_with_default(
        "Poster status",
        category_options.get("poster_status", ["Verified", "Non Verified", "No Registered"]),
        "Verified",
    )

    project_name_grouped = select_with_default(
        "Project name",
        load_project_options(),
        "Other"
    )

    st.divider()
    st.subheader("Amenities")

    air_con = st.checkbox("Air conditioner")
    digital_door_lock = st.checkbox("Digital door lock")
    furnished = st.checkbox("Furnished")
    hot_tub = st.checkbox("Hot tub")
    in_room_wifi = st.checkbox("In-room WiFi")
    kitchen_hood = st.checkbox("Kitchen hood")
    kitchen_stove = st.checkbox("Kitchen stove")
    phone = st.checkbox("Phone")
    refrigerator = st.checkbox("Refrigerator")
    tv = st.checkbox("TV")
    washer = st.checkbox("Washer")
    water_heater = st.checkbox("Water heater")


amenities = {
    "air_conditioner": int(air_con),
    "digital_door_lock": int(digital_door_lock),
    "furnished": int(furnished),
    "hot_tub": int(hot_tub),
    "in_room_wifi": int(in_room_wifi),
    "kitchen_hood": int(kitchen_hood),
    "kitchen_stove": int(kitchen_stove),
    "phone": int(phone),
    "refrigerator": int(refrigerator),
    "tv": int(tv),
    "washer": int(washer),
    "water_heater": int(water_heater),
}

amenity_count = sum(amenities.values())
kitchen_ready = amenities["kitchen_stove"] + amenities["kitchen_hood"] + amenities["refrigerator"]
move_in_ready = amenities["furnished"] + amenities["air_conditioner"] + amenities["refrigerator"] + amenities["washer"]
comfort_score = amenities["tv"] + amenities["in_room_wifi"] + amenities["air_conditioner"]

if popular_projects:
    is_popular_project = int(project_name_grouped in popular_projects)
else:
    is_popular_project = int(project_name_grouped != "Other")

input_df = pd.DataFrame(
    [
        {
            # "no_bedroom": no_bedroom,
            "room_size": room_size,
            # "floor": floor,
            "amenity_count": amenity_count,
            "kitchen_ready": kitchen_ready,
            "move_in_ready": move_in_ready,
            "comfort_score": comfort_score,
            "is_popular_project": is_popular_project,
            "poster_status": poster_status,
            "unit_type": unit_type,
            "project_name_grouped": project_name_grouped,
            "room_size_group": room_size_group(room_size),
        }
    ]
)
    
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Input summary")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("Prediction")

    if st.button("Predict Rental Price", type='primary', use_container_width=True):
        prediction = float(model.predict(input_df)[0])
        st.metric("Estimated monthly rental", f"฿{prediction:,.0f}")

        st.info(
            "This is a model estimate, not a guaranteed market price. "
            "Use it as a starting point for comparison."
        )