""" This script loads the original recipe dataset and processes it to create a final .csv for modelling. """

import kagglehub
import pandas as pd
import re

from kagglehub import KaggleDatasetAdapter


''' LOAD INITIAL DATASET '''
# Set file path
file_path = 'recipes.csv'

# Load the latest version
data = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "irkaal/foodcom-recipes-and-reviews",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

# Check raw data
print("RAW DATA:")
print(data.head())

# Drop unwanted variables
columns_to_keep = ['Name', 'RecipeCategory', 'RecipeIngredientParts', 'RecipeInstructions']
data = data[columns_to_keep]
print("\nWORKING DATA:")
print(data.head())


''' CLEAN DATASET '''
# Drop rows with missing data
data = data.dropna(subset=columns_to_keep)

# Streamline "Name" column
data['Name'] = data['Name'].str.lower().str.strip()

# Get all unique RecipeCategory values
unique_categories = data['RecipeCategory'].unique()
print("\nUNIQUE KEYWORDS:")

for category in unique_categories:
    print(category)

# Define generalized keyword groups for each target label
# There will be some cross over between categories, but is mostly generalized
category_keywords = {
    'Beverages': ['Beverages', 'Punch Beverage', 'Fruit', 'Smoothies', 'Shakes', 'Margarita'],

    'Dessert': ['Frozen Desserts', 'Pie', 'Dessert', 'Cheesecake', 'Bar Cookie', 'Scones', 'Drop Cookies', 'Apple', 'Candy', 
                'Gelatin', 'Tarts', 'Sweet', 'Chocolate Chip Cookies', 'Peanut Butter Pie', 'Bread Pudding', 'Lemon Cake', 'Ice Cream', 
                'Apple Pie', 'Key Lime Pie', 'Desserts Fruit', 'Birthday', 'Coconut Cream Pie'],

    'Breads': ['Yeast Breads', 'Breads', 'Quick Breads', 'Sourdough Breads', 'Buttermilk Biscuits', 'Wheat Bread', 'Baking'],

    'Soups': ['Clear Soup', 'Ham And Bean Soup', 'Bean Soup', 'Black Bean Soup', 'Potato Soup', 'Broccoli Soup', 'Soups Crock Pot', 
              'Mushroom Soup', 'Stocks', 'Chowders'],

    'Main Dish': ['Chicken Breast', 'Soy/Tofu', 'Chicken', 'Stew', 'Whole Chicken', 'High Protein', 
               'Brown Rice', 'Pork', 'Halibut', 'Meat', 'Lamb/Sheep', 'Very Low Carbs', 'Spaghetti', '< 30 Mins', 'Curries', 
               'Chicken Livers', '< 60 Mins', 'Savory Pies', 'Poultry', 'Steak', 'Lobster', 'Broil/Grill', 'Crab', 'Bass', 'Manicotti', 
               'Chicken Thigh & Leg', 'Lentil', 'Tuna', 'Crawfish', 'Beef Organ Meats', 'One Dish Meal', 'Veal', 'Orange Roughy', 
               'Mussels', 'Medium Grain Rice', 'Penne', 'Elk', 'Gumbo', 'Roast Beef', 'Perch', 'Rabbit', 'Kid Friendly', 'Whole Turkey', 
               'Meatloaf', 'Trout', 'Goose', 'Pasta Shells', 'Meatballs', 'Whole Duck', '< 4 Hours', 'Catfish', 'Duck Breasts', 'Stir Fry', 'Deer', 'Wild Game', 
               'Pheasant', 'No Shell Fish', 'Tilapia', 'Quail', 'Pressure Cooker', 'Squid', 'Plums', 'Mahi Mahi', 'Moose', 'Tempeh', 
               'Turkey Breasts', 'Duck', 'Pot Pie', 'Lemon', 'Toddler Friendly', 'Whitefish', 'Stove Top', 'Main Dish Casseroles', 
               'Pot Roast', 'Roast Beef Crock Pot', 'Chicken Crock Pot'],

    'Sides': ['Vegetable', 'Black Beans', 'Lactose Free', 'Oranges', 'Low Protein', 'Asian', 'Potato', 'Cheese', 'Beans', 'Pineapple', 
              'Rice', 'Pears', 'Cauliflower', 'White Rice', 'Onions', 'Corn', 'Long Grain Rice', 'Citrus', 'Berries', 'Peppers', 
              'Strawberry', 'Short Grain Rice', '< 15 Mins', 'Spicy', 'Oven', 'Microwave', 'Melons', 'Papaya', 'Potluck', 'Vegan', 
              'For Large Groups', 'Grains', 'Yam/Sweet Potato', 'Ham', 'Greens', 'Savory', 'Collard Greens', 'Refrigerator', 
              'Spinach', 'Tropical Fruits', 'Canning', 'Mango', 'Cherries', 'Chard', 'Octopus', 'Deep Fried', 'Beginner Cook', 
              'Egg Free', 'Gluten Free Appetizers', 'Macaroni and Cheese', 'Pumpkin', 'Beef Liver'],

    'Sauces': ['Sauces', 'Spreads', 'Jellies', 'Chutneys', 'Salad Dressings', 'Raspberries', 'Japanese', 'High Fiber', 'Ethiopian', 
               'Lime', 'Kiwifruit', 'Turkey Gravy', 'Spaghetti Sauce'],

    'Snacks': ['Lunch/Snacks', 'Dehydrator', 'Snacks Sweet'],
    
    'Breakfast': ['Breakfast', 'Peanut Butter', 'Brunch', 'Breakfast Eggs', 'Oatmeal', 'Breakfast Caseroles']
}

# Seperate out different cuisuine types for future feature engineering
cuisines = ['Southwestern U.S.',  'Brazilian',  'German', 'European', 'Hungarian', 'New Zealand',  'Indonesian',  'Mexican',  
            'Cajun', 'Russian', 'Vietnamese', 'Chinese', 'Native American',  'Polish', 'Moroccan', 'Creole', 'Hawaiian', 'Austrian', 
            'Swedish', 'Belgian', 'Australian', 'Cuban', 'Szechuan', 'Costa Rican',  'Welsh', 'Malaysian', 'Nigerian','Finnish', 
            'Nepalese', 'Hunan', 'Chilean', 'Cambodian', 'Sudanese', 'Peruvian', 'Somalian', 'Tex Mex', 'Greek', 
            'Southwest Asia (middle East)', 'Spanish', 'Dutch', 'Thai', 'Swiss', 'Canadian', 'Lebanese', 'Turkish', 'African', 
            'Scandinavian', 'Korean', 'Danish', 'Norwegian', 'Pennsylvania Dutch', 'Scottish', 'Cantonese', 'Portuguese', 'Filipino', 
            'Polynesian', 'South American', 'Venezuelan', 'Georgian', 'Palestinian', 'Icelandic', 'Iraqi', 'Puerto Rican', 'Honduran',
            'Indian']

# Categorize each recipe into one of the target labels
def map_recipe_category(raw_category, category_keywords):
    if pd.isna(raw_category):
        return "Other"
    
    raw_category = str(raw_category).lower()
    
    for main_category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword.lower() in raw_category:
                return main_category
            
    return "Other"

data['Recipe Category'] = data['RecipeCategory'].apply(lambda x: map_recipe_category(x, category_keywords))
data = data.drop(axis=1, columns=['RecipeCategory'])

data["Recipe Category"].value_counts()

# Process recipe ingredient list and instructions variables -- convert to plain text
def clean_recipe_text(text):
    if pd.isna(text):
        return ''
    
    text = re.sub(r'^c\(|\)$', '', text.strip()) # Remove "c(" and ")"
    text = text.replace('"', '').replace("'", "") # Remove quotes
    text = text.replace('.', '') # Remove periods
    text = text.replace(",", " ") # Replace commas with spaces
    text = text.lower() # Lowercase
    text = re.sub(r'\s+', ' ', text).strip() # Remove whitespace

    return text

data['Ingredients'] = data['RecipeIngredientParts'].apply(clean_recipe_text)
data['Instructions'] = data['RecipeInstructions'].apply(clean_recipe_text)
data = data.drop(axis=1, columns=['RecipeIngredientParts', 'RecipeInstructions'])

print("PROCESSED DATA:")
print(data.head())

# Save processed data to a .csv
data.to_csv('../data/processed_data.csv', index=False)
print(f"\nDataset successfully saved to '{'../data/processed_data.csv'}'")