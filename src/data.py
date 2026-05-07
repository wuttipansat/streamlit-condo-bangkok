import pandas as pd
from pathlib import Path
import re
import numpy as np

def load_data(path: str, config: dict) -> pd.DataFrame:
    df = pd.read_excel(Path(path))
    return processed_data(df, config)

def clean_room_size(value):

    if value is None or value == "":
        return np.nan

    value = str(value).lower().strip()

    nums = re.search(r"\d+\.?\d*", value)

    if nums:
        return float(nums[0])
    
    return np.nan

def clean_floor(value):

    if value is None or value == "":
        return np.nan
    
    value = str(value).lower().strip()

    if value in ["xx floor", "x floor", "xx", "-", "", "x", "nan"]:
        return np.nan
    
    value = value.replace("floor", "").strip()

    value = re.sub(r"(st|nd|rd|th)", "", value)

    value = value.replace("+", "").strip()

    if "-" in value:
        nums = re.findall(r"\d+", value)
        if len(nums) == 2:
            floor = (int(nums[0]) + int(nums[1])) / 2
            
            if floor <= 0:
                return np.nan

            return floor

    nums = re.findall(r"\d+", value)
    
    if len(nums) > 0:
        floor = int(nums[0])

        if floor <= 0:
            return np.nan
        
        return floor
    
    return np.nan

def clean_project_name(df: pd.DataFrame, n: int = 15):

    df = df.copy()

    df['project_name'] = df['project_name'].fillna("Unknown").astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    
    counts = df['project_name'].value_counts()
    
    valid = counts[counts >= n].index

    df['project_name_grouped'] = df['project_name'].where(
        df['project_name'].isin(valid),
        "Other"
    )

    df = df.drop(columns='project_name')

    return df

def clean_poster_status(df: pd.DataFrame):

    df = df.copy()
    df['poster_status'] = (df['poster_status']
                           .fillna("No Registered")
                           .astype(str)
                           .replace({
            "Verified member": "Verified",
            "Listing from Non verified member": "Non Verified"
        }))

    return df

def clean_no_bedroom(df: pd.DataFrame):
    
    df = df.copy()

    df['no_bedroom'] = df['no_bedroom'].astype(str).str.replace("room", "", regex=False).str.strip()

    df['no_bedroom'] = pd.to_numeric(df['no_bedroom'], errors='coerce')

    df['no_bedroom'] = df['no_bedroom'].fillna(0)

    return df    

def processed_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()

    selected_columns = config["columns"]["selected"] + config["columns"]["amenities"]
    amenity_columns = config["columns"]["amenities"]
    target = config["data"]["target"]
    min_project_freq = config["project"]["min_project_frequency"]

    selected_columns = [col for col in selected_columns if col in df.columns]
    df = df[selected_columns]

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

  
    df["room_size"] = df["room_size"].apply(clean_room_size)
    df = df[(df["room_size"] >= 10) & (df["room_size"] <= 1000)]


    df["floor"] = df["floor"].apply(clean_floor)
    df["floor"] = df["floor"].fillna(df["floor"].median())


    df = clean_project_name(df, n=min_project_freq)
    df["project_name_grouped"] = (
        df["project_name_grouped"]
        .fillna("Other")
        .astype(str)
        .str.strip()
    )


    df = clean_poster_status(df)
    df["poster_status"] = df["poster_status"].fillna("No Registered")


    df = clean_no_bedroom(df)
    df["no_bedroom"] = df["no_bedroom"].fillna(0)

    data = data.drop_duplicates()


    available_amenities = [col for col in amenity_columns if col in df.columns]

    for col in available_amenities:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = df[col].clip(0, 1)


    def safe_col(col_name: str):
        if col_name in df.columns:
            return df[col_name]
        return 0

    df["amenity_count"] = df[available_amenities].sum(axis=1)

    df["kitchen_ready"] = (
        safe_col("kitchen_stove")
        + safe_col("kitchen_hood")
        + safe_col("refrigerator")
    )

    df["move_in_ready"] = (
        safe_col("furnished")
        + safe_col("air_conditioner")
        + safe_col("refrigerator")
        + safe_col("washer")
    )

    df["comfort_score"] = (
        safe_col("tv")
        + safe_col("in_room_wifi")
        + safe_col("air_conditioner")
    )

    df["room_size_group"] = pd.cut(
        df["room_size"],
        bins=[0, 25, 35, 50, 80, 150, np.inf],
        labels=["XS", "S", "M", "L", "XL", "XXL"]
    )

    proj_counts = df["project_name_grouped"].value_counts()

    df["project_counts"] = df["project_name_grouped"].map(proj_counts)

    df["is_popular_project"] = (
        (df["project_counts"] >= 100)
        & (df["project_name_grouped"] != "Other")
    ).astype(int)


    df = df.drop(columns=available_amenities)
    df = df.drop(columns="project_counts")



    return df
