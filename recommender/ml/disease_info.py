"""
Static disease information for the AgroSmart result page.

Keys match label_map.json EXACTLY: "Crop__Disease_Name" (e.g.
"Maize__Common_Rust"), using the same crop names and folder/class names
finalized during dataset construction. All 29 classes across 7 crops are
covered. Each section (symptoms, causes, prevention, treatment) is a list
of 3-5 short bullet points.

This file has no dependency on the CNN, the model, or any training code —
pure display data, safe to edit anytime without touching the prediction
pipeline.
"""

DISEASE_INFO = {

    # ==================== Maize (4 classes) ====================
    "Maize__Healthy": {
        "disease_name": "Healthy",
        "symptoms": [
            "Uniform green leaf color with no spots or lesions",
            "Normal leaf shape and size for the growth stage",
            "No wilting, curling, or premature yellowing",
        ],
        "causes": [
            "N/A — the plant shows no signs of disease",
        ],
        "prevention": [
            "Maintain balanced NPK fertilization",
            "Ensure proper plant spacing for airflow",
            "Monitor fields regularly for early signs of stress",
            "Rotate crops each season to reduce pathogen buildup",
        ],
        "treatment": [
            "No treatment needed",
        ],
    },
    "Maize__Common_Rust": {
        "disease_name": "Common Rust",
        "symptoms": [
            "Small, circular to elongated reddish-brown pustules on both leaf surfaces",
            "Pustules turn dark brown/black as the plant matures",
            "Severe cases show extensive yellowing and leaf death",
        ],
        "causes": [
            "Fungus Puccinia sorghi",
            "Favored by cool temperatures (16-23°C)",
            "Spreads rapidly under high humidity or prolonged leaf wetness",
            "Windborne spore dispersal between fields",
        ],
        "prevention": [
            "Plant rust-resistant hybrids",
            "Rotate crops to reduce spore carryover",
            "Avoid excessive nitrogen, which promotes dense, humid canopies",
            "Space plants adequately for airflow",
        ],
        "treatment": [
            "Apply a triazole or strobilurin-based foliar fungicide at first sign of pustules",
            "Repeat application on a 10-14 day interval if pressure remains high",
            "Remove severely infected residue after harvest",
        ],
    },
    "Maize__Gray_Leaf_Spot": {
        "disease_name": "Gray Leaf Spot",
        "symptoms": [
            "Narrow, rectangular tan-to-gray lesions running parallel to leaf veins",
            "Lesions merge to blight large sections of the leaf",
            "Lower leaves affected first, moving upward over time",
        ],
        "causes": [
            "Fungus Cercospora zeae-maydis",
            "Thrives in warm, humid conditions",
            "Worse under reduced tillage with crop residue left on the surface",
            "Extended leaf wetness accelerates spread",
        ],
        "prevention": [
            "Rotate with non-host crops",
            "Till under infected residue to reduce fungal survival",
            "Choose resistant hybrids where available",
            "Avoid overly dense planting",
        ],
        "treatment": [
            "Foliar fungicide application at early disease onset",
            "Repeat treatment if conditions remain warm and humid",
            "Prioritize susceptible hybrids for fungicide programs",
        ],
    },
    "Maize__Northern_Leaf_Blight": {
        "disease_name": "Northern Leaf Blight",
        "symptoms": [
            "Long, cigar-shaped grayish-green to tan lesions (2.5-15 cm)",
            "Lesions often start on lower leaves and move upward",
            "Severe infection can blight most of the leaf canopy",
        ],
        "causes": [
            "Fungus Exserohilum turcicum",
            "Favored by moderate temperatures (18-27°C)",
            "Requires extended periods of leaf wetness to infect",
            "Survives in crop residue between seasons",
        ],
        "prevention": [
            "Use resistant hybrids",
            "Rotate crops to break the disease cycle",
            "Manage/till residue to reduce fungal carryover",
            "Avoid late planting in high-risk regions",
        ],
        "treatment": [
            "Apply fungicide early, before lesions spread extensively",
            "Prioritize seed or high-value fields for treatment",
            "Monitor weather — treat proactively before extended wet periods",
        ],
    },

    # ==================== Potato (3 classes) ====================
    "Potato__Healthy": {
        "disease_name": "Healthy",
        "symptoms": [
            "Leaves green and undamaged",
            "No spots, wilting, or lesions",
            "Normal growth rate and leaf size",
        ],
        "causes": [
            "N/A — the plant shows no signs of disease",
        ],
        "prevention": [
            "Maintain good field hygiene",
            "Use proper irrigation scheduling",
            "Apply balanced fertilization",
            "Rotate with non-host crops",
        ],
        "treatment": [
            "No treatment needed",
        ],
    },
    "Potato__Early_Blight": {
        "disease_name": "Early Blight",
        "symptoms": [
            "Dark brown spots with concentric rings (target-like pattern)",
            "Usually starts on older, lower leaves first",
            "Yellowing around lesions, followed by leaf drop in severe cases",
        ],
        "causes": [
            "Fungus Alternaria solani",
            "Favored by warm temperatures with alternating wet/dry periods",
            "More severe on nutrient-deficient or stressed plants",
            "Survives in soil and infected debris between seasons",
        ],
        "prevention": [
            "Maintain balanced fertility, especially nitrogen",
            "Avoid plant stress from drought or nutrient deficiency",
            "Rotate crops and remove infected debris after harvest",
            "Use certified disease-free seed potatoes",
        ],
        "treatment": [
            "Apply a protectant fungicide (e.g. chlorothalonil or copper-based) at first symptoms",
            "Repeat on a 7-10 day interval during active disease pressure",
            "Remove and destroy heavily infected foliage",
        ],
    },
    "Potato__Late_Blight": {
        "disease_name": "Late Blight",
        "symptoms": [
            "Water-soaked, pale-to-dark green lesions that rapidly turn brown/black",
            "White fungal growth on leaf undersides in humid conditions",
            "Can destroy entire fields within days under favorable weather",
        ],
        "causes": [
            "Oomycete Phytophthora infestans",
            "Spreads explosively in cool, wet, humid weather",
            "Spores carried by wind and rain over long distances",
            "Survives in infected tubers and volunteer plants",
        ],
        "prevention": [
            "Use certified disease-free seed potatoes",
            "Ensure good field drainage and airflow",
            "Avoid overhead irrigation late in the day",
            "Monitor weather forecasts for high-risk conditions",
        ],
        "treatment": [
            "Apply systemic fungicides immediately upon detection",
            "Destroy severely infected plants to prevent further spread",
            "Harvest early if blight pressure is severe, to protect tubers",
        ],
    },

    # ==================== Rice (4 classes) ====================
    "Rice__Healthy": {
        "disease_name": "Healthy",
        "symptoms": [
            "Leaves green and undamaged",
            "No lesions, spots, or discoloration",
            "Normal tillering and growth",
        ],
        "causes": [
            "N/A — the plant shows no signs of disease",
        ],
        "prevention": [
            "Maintain proper water management",
            "Apply balanced fertilization",
            "Use certified healthy seed",
        ],
        "treatment": [
            "No treatment needed",
        ],
    },
    "Rice__Bacterial_Leaf_Blight": {
        "disease_name": "Bacterial Leaf Blight",
        "symptoms": [
            "Water-soaked streaks near leaf tips/margins",
            "Streaks turn yellow, then white/gray with a wavy margin",
            "Can cause whole-leaf wilting in seedlings (kresek phase)",
        ],
        "causes": [
            "Bacterium Xanthomonas oryzae pv. oryzae",
            "Spreads through irrigation water and wind-driven rain",
            "Contaminated tools and infected seed carry the pathogen",
            "Favored by warm, humid weather",
        ],
        "prevention": [
            "Use resistant varieties",
            "Avoid excess nitrogen fertilization",
            "Ensure clean irrigation water sources",
            "Avoid field work when leaves are wet",
        ],
        "treatment": [
            "No fully effective chemical cure once established",
            "Copper-based bactericides may reduce further spread",
            "Remove and destroy severely infected plants",
        ],
    },
    "Rice__Brown_Spot": {
        "disease_name": "Brown Spot",
        "symptoms": [
            "Small, circular to oval brown spots with gray/white centers",
            "Appears on leaves and sometimes on grains",
            "Can cause significant yield loss in nutrient-poor soils",
        ],
        "causes": [
            "Fungus Cochliobolus miyabeanus (Bipolaris oryzae)",
            "Linked to nutrient-deficient soils, especially low potassium",
            "Worsened by water stress and poor seed quality",
        ],
        "prevention": [
            "Improve soil fertility, particularly potassium and silicon",
            "Use healthy, certified seed",
            "Avoid water stress during critical growth stages",
        ],
        "treatment": [
            "Fungicide seed treatment before planting",
            "Foliar fungicide sprays in severe outbreaks",
            "Correct underlying soil nutrient deficiencies",
        ],
    },
    "Rice__Blast": {
        "disease_name": "Blast",
        "symptoms": [
            "Diamond/spindle-shaped lesions with gray centers and brown margins",
            "Can affect neck and panicle, causing grain loss (neck blast)",
            "Severe infection kills entire leaves",
        ],
        "causes": [
            "Fungus Magnaporthe oryzae",
            "Favored by high humidity and extended leaf wetness",
            "Worsened by excessive nitrogen fertilization",
            "Spreads via airborne spores",
        ],
        "prevention": [
            "Use resistant varieties",
            "Avoid excess nitrogen application",
            "Maintain proper plant spacing for airflow",
            "Manage water levels carefully during susceptible stages",
        ],
        "treatment": [
            "Apply a systemic fungicide (e.g. tricyclazole) at early symptom onset",
            "Time treatment before panicle emergence if neck blast risk is high",
            "Repeat application if humid conditions persist",
        ],
    },

    # ==================== Wheat (3 classes) ====================
    "Wheat__Healthy": {
        "disease_name": "Healthy",
        "symptoms": [
            "Leaves green and undamaged",
            "No rust pustules or lesions",
            "Normal tillering and canopy development",
        ],
        "causes": [
            "N/A — the plant shows no signs of disease",
        ],
        "prevention": [
            "Maintain balanced fertilization",
            "Practice proper field sanitation",
            "Rotate with non-cereal crops",
        ],
        "treatment": [
            "No treatment needed",
        ],
    },
    "Wheat__Stripe_Rust": {
        "disease_name": "Stripe Rust",
        "symptoms": [
            "Bright yellow-orange pustules in narrow stripes along leaf veins",
            "Stripes may merge, covering large leaf areas",
            "Severe infection reduces grain fill and yield",
        ],
        "causes": [
            "Fungus Puccinia striiformis",
            "Favored by cool temperatures (10-15°C)",
            "Spreads via windborne spores over long distances",
            "High humidity accelerates infection",
        ],
        "prevention": [
            "Plant resistant wheat varieties",
            "Monitor fields closely during cool, humid seasons",
            "Avoid excessive nitrogen which increases susceptibility",
        ],
        "treatment": [
            "Apply a triazole-based foliar fungicide at early detection",
            "Treat before pustules spread extensively across the canopy",
            "Repeat if cool, humid conditions persist",
        ],
    },
    "Wheat__Septoria": {
        "disease_name": "Septoria Leaf Blotch",
        "symptoms": [
            "Irregular tan/brown blotches on leaves",
            "Tiny black fruiting bodies (pycnidia) visible within lesions",
            "Often starts on lower leaves, moving upward",
        ],
        "causes": [
            "Fungus Zymoseptoria tritici (formerly Septoria tritici)",
            "Spreads via rain-splash from infected residue",
            "Favored by cool, wet conditions",
        ],
        "prevention": [
            "Rotate crops to reduce residue-borne inoculum",
            "Bury or remove infected crop residue",
            "Use resistant varieties where available",
        ],
        "treatment": [
            "Apply foliar fungicide, particularly at the flag-leaf stage",
            "Protect yield-critical upper leaves with timely spraying",
            "Repeat treatment if wet weather continues",
        ],
    },

    # ==================== Sugarcane (3 classes) ====================
    "Sugarcane__Healthy": {
        "disease_name": "Healthy",
        "symptoms": [
            "Leaves green and undamaged",
            "No internal discoloration or reddened tissue",
            "Normal stalk growth and development",
        ],
        "causes": [
            "N/A — the plant shows no signs of disease",
        ],
        "prevention": [
            "Maintain proper irrigation and fertilization",
            "Use disease-free planting material",
            "Avoid waterlogging",
        ],
        "treatment": [
            "No treatment needed",
        ],
    },
    "Sugarcane__Red_Rot": {
        "disease_name": "Red Rot",
        "symptoms": [
            "Reddening of internal stalk tissue with white patches (visible when cut open)",
            "Yellowing and drying of leaves",
            "Foul smell from infected stalks in advanced cases",
        ],
        "causes": [
            "Fungus Colletotrichum falcatum",
            "Spreads through infected seed cane",
            "Favored by waterlogging and wounds from field operations",
        ],
        "prevention": [
            "Use disease-free, resistant seed cane varieties",
            "Avoid waterlogging in fields",
            "Rotate with non-host crops",
            "Disinfect cutting tools between plants",
        ],
        "treatment": [
            "No effective in-field cure once internal rot sets in",
            "Remove and destroy infected clumps to prevent spread",
            "Treat seed cane with hot water or fungicide before planting",
        ],
    },
    "Sugarcane__Red_Rust": {
        "disease_name": "Red Rust",
        "symptoms": [
            "Small reddish-brown pustules on leaf surfaces",
            "Pustules often surrounded by a yellow halo",
            "Merges to cover large leaf areas in severe cases",
        ],
        "causes": [
            "Rust fungi (Puccinia species)",
            "Favored by high humidity and moderate temperatures",
            "Spreads via windborne spores",
        ],
        "prevention": [
            "Use resistant varieties",
            "Ensure balanced fertilization to reduce plant stress",
            "Maintain adequate plant spacing for airflow",
        ],
        "treatment": [
            "Fungicide sprays can control severe outbreaks",
            "Resistant variety selection is the primary long-term control",
        ],
    },

    # ==================== Mango (8 classes) ====================
    "Mango__Healthy": {
        "disease_name": "Healthy",
        "symptoms": [
            "Leaves green and undamaged",
            "No spots, lesions, or wilting",
            "Normal flowering and fruit development",
        ],
        "causes": [
            "N/A — the plant shows no signs of disease",
        ],
        "prevention": [
            "Maintain proper pruning and drainage",
            "Practice good orchard sanitation",
            "Monitor during flowering season",
        ],
        "treatment": [
            "No treatment needed",
        ],
    },
    "Mango__Anthracnose": {
        "disease_name": "Anthracnose",
        "symptoms": [
            "Dark brown to black irregular spots on leaves, flowers, and fruit",
            "Spots enlarge and merge, causing leaf drop and fruit rot",
            "Severe cases cause flower blight and reduced fruit set",
        ],
        "causes": [
            "Fungus Colletotrichum gloeosporioides",
            "Thrives in warm, humid weather",
            "Especially active during flowering and fruit development",
            "Spreads via rain-splash and wind",
        ],
        "prevention": [
            "Prune for good air circulation",
            "Remove fallen debris and infected plant material",
            "Avoid overhead irrigation during flowering",
        ],
        "treatment": [
            "Apply copper-based or strobilurin fungicide at flowering",
            "Repeat treatment before fruit ripening",
            "Remove and destroy infected plant parts",
        ],
    },
    "Mango__Bacterial_Canker": {
        "disease_name": "Bacterial Canker",
        "symptoms": [
            "Water-soaked, angular lesions on leaves that turn dark brown/black",
            "Raised corky cankers on stems and fruit",
            "Fruit cracking and premature drop in severe cases",
        ],
        "causes": [
            "Bacterium Xanthomonas campestris pv. mangiferaeindicae",
            "Spreads via wind-driven rain",
            "Enters through wounds, especially after storm damage",
            "Contaminated tools spread infection between trees",
        ],
        "prevention": [
            "Use disease-free planting material",
            "Avoid working in orchards when wet",
            "Disinfect pruning tools between trees",
        ],
        "treatment": [
            "Copper-based bactericide sprays to reduce spread",
            "Prune and destroy severely infected branches",
        ],
    },
    "Mango__Cutting_Weevil": {
        "disease_name": "Cutting Weevil (Leaf/Shoot Damage)",
        "symptoms": [
            "Notched, chewed leaf margins",
            "Damaged young shoots and petioles",
            "Visible feeding marks from adult weevils",
        ],
        "causes": [
            "Weevil pests (e.g. Deporaus marginatus)",
            "Adults feed on and cut young leaves/shoots, often to lay eggs",
            "More active during new flush growth",
        ],
        "prevention": [
            "Monitor young flush growth closely",
            "Remove and destroy fallen damaged leaves",
            "Encourage natural predators in the orchard",
        ],
        "treatment": [
            "Targeted insecticide application during new flush emergence",
            "Consult local agricultural extension for approved products",
        ],
    },
    "Mango__Die_Back": {
        "disease_name": "Die Back",
        "symptoms": [
            "Progressive drying and death of twigs and branches from the tip",
            "Dark discoloration of bark",
            "Reduced flowering and fruiting on affected branches",
        ],
        "causes": [
            "Fungi such as Botryosphaeria/Lasiodiplodia species",
            "Enters through wounds, pruning cuts, or stressed tissue",
            "Worsened by drought stress or nutrient deficiency",
        ],
        "prevention": [
            "Prune with clean, disinfected tools",
            "Seal large pruning cuts",
            "Avoid plant stress from drought or poor nutrition",
        ],
        "treatment": [
            "Prune out and destroy affected branches well below visible symptoms",
            "Apply fungicidal paste to fresh cuts",
            "Improve overall tree vigor through proper irrigation and feeding",
        ],
    },
    "Mango__Gall_Midge": {
        "disease_name": "Gall Midge",
        "symptoms": [
            "Small raised bumps (galls) on leaves, shoots, or flower panicles",
            "Galls form around developing larvae inside plant tissue",
            "Distorted or stunted new growth",
        ],
        "causes": [
            "Fly larvae (Procontarinia species)",
            "Eggs laid in young plant tissue by adult flies",
            "Galls form as protective structures around developing larvae",
        ],
        "prevention": [
            "Remove and destroy fallen infested leaves/flowers",
            "Break the pest life cycle through sanitation",
            "Monitor closely during flowering",
        ],
        "treatment": [
            "Insecticide application timed to flowering/new flush stages",
            "Soil treatment may target pupating larvae",
        ],
    },
    "Mango__Powdery_Mildew": {
        "disease_name": "Powdery Mildew",
        "symptoms": [
            "White, powdery fungal growth on leaves, flowers, and young fruit",
            "Flower drop and fruit deformation in severe cases",
            "Stunted growth of new shoots",
        ],
        "causes": [
            "Fungus Oidium mangiferae",
            "Favored by cool nights, warm days, and high humidity without rainfall",
            "Dew-heavy conditions accelerate spread",
        ],
        "prevention": [
            "Prune for airflow, avoid dense canopy growth",
            "Monitor closely during flowering season",
            "Avoid excessive nitrogen fertilization",
        ],
        "treatment": [
            "Sulfur-based or systemic fungicide sprays at first bloom",
            "Repeat through fruit set for continued protection",
        ],
    },
    "Mango__Sooty_Mould": {
        "disease_name": "Sooty Mould",
        "symptoms": [
            "Black, soot-like fungal coating on leaf and fruit surfaces",
            "Reduced photosynthesis due to blocked sunlight",
            "Doesn't directly infect plant tissue but signals insect infestation",
        ],
        "causes": [
            "Fungi (e.g. Capnodium species) growing on insect honeydew",
            "Secondary to sap-sucking insects like aphids, scale, or mealybugs",
            "Not a direct plant pathogen — always linked to an insect problem",
        ],
        "prevention": [
            "Control underlying insect pests (aphids, scale, mealybugs)",
            "Monitor for early signs of insect infestation",
        ],
        "treatment": [
            "Wash affected leaves/fruit to remove mould buildup",
            "Apply insecticide or horticultural oil to control honeydew-producing insects",
            "Mould clears naturally once insects are managed",
        ],
    },

    # ==================== Banana (4 classes) ====================
    "Banana__Healthy": {
        "disease_name": "Healthy",
        "symptoms": [
            "Leaves green and undamaged",
            "No spots, streaks, or wilting",
            "Normal growth and fruit development",
        ],
        "causes": [
            "N/A — the plant shows no signs of disease",
        ],
        "prevention": [
            "Maintain proper spacing and drainage",
            "Use disease-free planting material",
            "Monitor plantation regularly",
        ],
        "treatment": [
            "No treatment needed",
        ],
    },
    "Banana__Sigatoka": {
        "disease_name": "Sigatoka Leaf Spot",
        "symptoms": [
            "Small yellow-brown streaks that develop into elongated brown/black spots",
            "Yellow halos around lesions",
            "Spots merge and kill large sections of the leaf",
        ],
        "causes": [
            "Fungus Mycosphaerella species (Black or Yellow Sigatoka)",
            "Spreads via windborne spores",
            "Favored by warm, humid, rainy conditions",
        ],
        "prevention": [
            "Ensure good plantation drainage and spacing for airflow",
            "Remove and destroy heavily infected leaves",
            "Avoid planting in poorly ventilated, low-lying areas",
        ],
        "treatment": [
            "Regular fungicide spray programs (e.g. propiconazole-based products)",
            "Common in commercial plantations to manage ongoing outbreaks",
        ],
    },
    "Banana__Cordana": {
        "disease_name": "Cordana Leaf Spot",
        "symptoms": [
            "Oval, light-brown to tan lesions with a distinct yellow halo",
            "Generally less aggressive than Sigatoka",
            "Slower-spreading, often on older or damaged leaves",
        ],
        "causes": [
            "Fungus Cordana musae",
            "Favored by high humidity",
            "Often affects older, damaged, or stressed leaves",
        ],
        "prevention": [
            "Avoid leaf wounding during handling",
            "Maintain good field sanitation by removing dead leaf material",
        ],
        "treatment": [
            "Usually not severe enough to require chemical treatment",
            "Fungicide can be applied if damage becomes extensive",
        ],
    },
    "Banana__Pestalotiopsis": {
        "disease_name": "Pestalotiopsis Leaf Spot",
        "symptoms": [
            "Grayish-brown spots with dark borders",
            "Often appears at leaf tips or margins",
            "Tiny black fungal fruiting structures sometimes visible",
        ],
        "causes": [
            "Fungi in the Pestalotiopsis genus",
            "Typically a secondary/opportunistic pathogen",
            "Infects already stressed or wounded leaf tissue",
        ],
        "prevention": [
            "Minimize leaf wounding during handling",
            "Maintain plant vigor through balanced nutrition",
            "Remove severely affected leaves",
        ],
        "treatment": [
            "Fungicide treatment rarely needed alone",
            "Addressing underlying plant stress usually resolves the issue",
        ],
    },
}


FALLBACK_INFO = {
    "disease_name": "Information not available.",
    "symptoms": ["Information not available."],
    "causes": ["Information not available."],
    "prevention": ["Information not available."],
    "treatment": ["Information not available."],
}


def get_disease_info(crop, disease):
    """
    crop: e.g. "Maize"  (from predict_disease()'s "crop" field)
    disease: e.g. "Common Rust"  (from predict_disease()'s "disease" field,
             spaces instead of underscores)

    Rebuilds the "Crop__Disease_Name" key to match DISEASE_INFO and
    label_map.json's exact format, then looks it up. Falls back to
    "Information not available." for anything not found.
    """
    key = f"{crop}__{disease.replace(' ', '_')}"
    return DISEASE_INFO.get(key, FALLBACK_INFO)


def get_disease_info_by_label(label):
    """
    Alternative lookup if you have the raw label_map.json value directly
    (e.g. "Maize__Common_Rust") instead of separate crop/disease strings.
    """
    return DISEASE_INFO.get(label, FALLBACK_INFO)