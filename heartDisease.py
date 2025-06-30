import pandas as pd

data=pd.read_csv("/Volumes/SSD/data.csv")

print(data.head())

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

x=data.drop(columns=["target"], axis=1)
y=data["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# visualizing the decesion tree

from sklearn.tree import export_graphviz
import graphviz

point=export_graphviz(model, out_file=None, 
                       feature_names=x.columns,  
                       class_names=["no problem", "problem"],  
                       filled=True, rounded=True,  
                       special_characters=True)

graph = graphviz.Source(point)  
graph.render("decision_tree")  
graph.view()  


# controlling overfitting with max depth
dt_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_pruned.fit(x_train, y_train)

y_pred_pruned = dt_pruned.predict(x_test)

print("Pruned Decision Tree Accuracy:", accuracy_score(y_test, y_pred_pruned))
print(classification_report(y_test, y_pred_pruned))
#  after pruning the f1 score is decreased but the model is less complex and more generalizable

# tree after pruning 
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(
    dt_pruned,
    out_file=None,
    feature_names=x.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("pruned_tree")  
graph.view() 



import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

randpm_f= RandomForestClassifier(n_estimators=100, random_state=42)
randpm_f.fit(x_train, y_train)

y_pred_rf = randpm_f.predict(x_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))



# cross validation from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(randpm_f, x, y, cv=5)
print("Cross-Validation Accuracy:", cv_scores.mean())



# interpret feature importances
importances = randpm_f.feature_importances_
feature_names = x.columns


df=pd.DataFrame({'Feature': feature_names, 'Importance': importances})
df = df.sort_values(by='Importance', ascending=False,inplace=True)

sns.barplot(x='Importance', y='Feature', data=df)
plt.title('Feature Importances')
plt.show()





