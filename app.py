import streamlit as st
import pandas as pd
import joblib

# Page Setup
st.set_page_config(page_title="Starbucks Decision Tree 2.0", page_icon="☕", layout="wide")

# Load Saved Files
@st.cache_resource
def load_assets():
    model = joblib.load("starbucks_dt_model.pkl")
    encoder = joblib.load("encoder.pkl")
    menu_data = joblib.load("menu_data.pkl")
    return model, encoder, menu_data

model, encoder, menu_data = load_assets()

# Header Section
st.title("☕ Starbucks Decision Tree 2.0")
st.markdown("Apni pasandida **Starbucks Drink** select karein, **Customizations** add karein, aur **Machine Learning (Decision Tree)** ki madad se real-time nutrition check karein!")
st.divider()

# Layout: Left Column (Selection & Customization) | Right Column (Results)
col_left, col_right = st.columns([1.2, 1.5], gap="large")

with col_left:
    st.subheader("1️⃣ Drink Select Karein")
    category = st.selectbox("Category:", list(menu_data.keys()))
    beverage = st.selectbox("Coffee / Drink:", menu_data[category])
    
    st.subheader("2️⃣ Base Size & Milk")
    c1, c2 = st.columns(2)
    with c1:
        size = st.selectbox("Cup Size:", ["Short", "Tall", "Grande", "Venti", "Solo", "Doppio"])
    with c2:
        milk = st.selectbox("Milk Option:", ["Nonfat Milk", "2% Milk", "Whole Milk", "Soymilk", "No Milk / Black"])

    st.subheader("3️⃣ Extra Customizations")
    sugar_pumps = st.slider("Extra Sugar / Flavoured Syrup (Pumps):", 0, 10, 0, help="1 pump ≈ 20 Calories aur 5g Sugar")
    espresso_shots = st.slider("Extra Espresso Shots:", 0, 4, 0, help="1 shot ≈ 5 Calories aur 75mg Caffeine")
    whipped_cream = st.checkbox("Add Whipped Cream (+70 Cal, +7g Fat)")
    extra_drizzle = st.checkbox("Add Caramel / Mocha Drizzle (+15 Cal, +3g Sugar)")

with col_right:
    st.subheader("📊 Predicted Nutrition Facts")
    
    # Input DataFrame tayyar karein
    input_df = pd.DataFrame([[category, beverage, size, milk]], 
                            columns=['Beverage_category', 'Beverage', 'Size', 'Milk'])
    
    # Encode aur Predict karein
    encoded_input = encoder.transform(input_df)
    base_pred = model.predict(encoded_input)[0]
    
    # Base predictions extract karein
    calories = base_pred[0]
    fat = base_pred[1]
    carbs = base_pred[2]
    sugar = base_pred[3]
    protein = base_pred[4]
    caffeine = base_pred[5]
    
    # Customizations ki values add karein
    calories += (sugar_pumps * 20) + (espresso_shots * 5) + (70 if whipped_cream else 0) + (15 if extra_drizzle else 0)
    fat += (7.0 if whipped_cream else 0) + (0.5 if extra_drizzle else 0)
    carbs += (sugar_pumps * 5) + (1 if espresso_shots else 0) + (2 if whipped_cream else 0) + (3 if extra_drizzle else 0)
    sugar += (sugar_pumps * 5) + (2 if whipped_cream else 0) + (3 if extra_drizzle else 0)
    protein += (1.0 if whipped_cream else 0)
    caffeine += (espresso_shots * 75)

    # Summary Banner
    st.info(f"**Selected:** {size} {beverage} ({milk}) | **Extras:** {sugar_pumps} Syrup Pumps, {espresso_shots} Extra Shots")

    # Metrics Cards (Row 1)
    m1, m2, m3 = st.columns(3)
    m1.metric("🔥 Calories", f"{max(0, calories):.0f} kcal")
    m2.metric("🥑 Total Fat", f"{max(0, fat):.1f} g")
    m3.metric("💪 Protein", f"{max(0, protein):.1f} g")

    # Metrics Cards (Row 2)
    m4, m5, m6 = st.columns(3)
    m4.metric("🍬 Sugars", f"{max(0, sugar):.1f} g")
    m5.metric("🍞 Carbs", f"{max(0, carbs):.1f} g")
    m6.metric("⚡ Caffeine", f"{max(0, caffeine):.0f} mg")

    # Visual Progress Bars (Daily Value % estimate)
    st.markdown("#### Daily Intake Contribution (Approx.)")
    st.write(f"**Calories ({calories:.0f} / 2000 kcal)**")
    st.progress(min(1.0, max(0.0, calories / 2000)))
    
    st.write(f"**Sugar ({sugar:.1f} / 50 g)**")
    st.progress(min(1.0, max(0.0, sugar / 50)))

    st.write(f"**Caffeine ({caffeine:.0f} / 400 mg)**")
    st.progress(min(1.0, max(0.0, caffeine / 400)))
