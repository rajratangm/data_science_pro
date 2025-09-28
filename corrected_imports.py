from data_science_pro import pipeline
import warnings
warnings.filterwarnings("ignore")


flow.input_data(r"C:\Users\Rajratan.More\Downloads\titanic.csv", target_col='Survived')
# flow.api()
# report = flow.report()
# suggestions = flow.suggestions("How can I improve my model accuracy?")

preprocessing_actions = [
    ('fill_na', 'Handle missing values'),
    ('drop_constant', 'Remove constant columns'),
    ('drop_high_na', 'Remove columns with >50% missing values'),
    ('encode_categorical', 'Encode categorical variables'),
    ('scale_numeric', 'Scale numeric features'),
    ('drop_duplicates', 'Remove duplicate rows'),
    ('feature_gen', 'Generate interaction features')
]
print(flow.suggestions("How can I improve my model accuracy?"))
for action, description in preprocessing_actions:
    print(f"\n🔄 {description}...")
    result = flow.apply_action(action)
    print(f"✅ {action} completed")
    print(f"   Data shape after {action}: {flow.data.shape}")

print(flow.suggestions("How can I improve my model accuracy?"))
