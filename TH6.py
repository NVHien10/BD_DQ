{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1sV8ItxEBPf-xjJi7Ikbhc6oZeA_SxQaS",
      "authorship_tag": "ABX9TyPwU5QiB5u85mSBMTT62zx9",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/NVHien10/BD_DQ/blob/main/TH6.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Ka8nIOjIKDtU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Cau 1"
      ],
      "metadata": {
        "id": "6vG1XgDhJVr8"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9DcPLhUa_wZc"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.tree import export_text\n",
        "from sklearn.tree import export_graphviz\n",
        "import graphviz\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tokenize import Special\n",
        "#Doc du lieu\n",
        "data = pd.read_excel(\"/content/drive/MyDrive/Khai Pha Du Lieu/TH6/Collected_Hr_data_performances.xls\")\n",
        "\n",
        "#chon cac cot dac trung va cot nhan\n",
        "features = data[[\"Age\", \"Gender\", \"MaritalStatus\", \"EducationLevel\", \"EducationBackground\",\n",
        "                 \"JobRole\", \"EnvironmentStatisfaction\", \"RelationshipStatisfaction\",\n",
        "                 \"WorkLifeBalance\",\"TotalWorkExperienceYears\", \"ExperienceYearsInCurrentRole\"]]\n",
        "\n",
        "labels = data[\"PerformanceResult\"]\n",
        "\n",
        "#chuyen doi cac cot du lieu dang van ban thanh dang so (thuc hien mo hinh huan luyen)\n",
        "features = pd.get_dummies(features)\n",
        "\n",
        "#Thay the cac ki tu k hop le, dac biet bang khoang trang\n",
        "features.columns = features.columns.str.replace('[^a-zA-Z0-9]',' ',regex = True)\n",
        "\n",
        "#Tach du lieu thanh hai tap train va Test\n",
        "train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)\n",
        "\n",
        "#xay dung mo hinh cay quyet dinh\n",
        "param_grid ={\n",
        "    'criterion': ['gini','entropy'],\n",
        "    'splitter' : ['best', ' random'],\n",
        "    'max_depth' : [None, 10, 20, 30],\n",
        "    'min_simples_split': [2,5,10],\n",
        "    'min_simples_leaf': [1,2,4]\n",
        "}\n",
        "\n",
        "#Tao mo hinh cay quyet dinh (j48)\n",
        "clf = DecisionTreeClassifier()\n",
        "\n",
        "#su dung gridSearch CV de thu nghiem cac tham so va lua chon mo hinh tot nhat\n",
        "grid_search = GridSearchCV(clf,param_grid, cv = 5, scoring = \"accuracy\")\n",
        "grid_search.fit(train_features, train_labels)\n",
        "\n",
        "#lua chon mo hinh tot nhat sau thu nghiem\n",
        "best_clf = grid_search.best_estimator_\n",
        "\n",
        "#in ra cac thong bao tot nhat cua mo hinh\n",
        "print(\"best Parameters:\", grid_search.best_params_)\n",
        "print(\"Best Accuracy: \", grid_search.best_score_)\n",
        "\n",
        "#in cac luat duoc hoc tu cay quyet dinh\n",
        "\n",
        "#Tao bieu dien do thi cho cay quyet dinh\n",
        "dot_data = export_graphviz(best_clf, outfile = None, feature_names=list(train_features.columns),\n",
        "                           class_names = best_clf.class_, filled = True, rounded= True,\n",
        "                           special_characters=True)\n",
        "# Hien thi do thi\n",
        "graph = graphviz.Source(dot_data)\n",
        "graph.render(fileneme= ' ', format = 'png', cleanup=True)\n",
        "graph.view()\n",
        "\n"
      ],
      "metadata": {
        "id": "LDLFGVJNJKxE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#du doan ket qua tren tap test bang mo hinh tot nhat da lua chon\n",
        "predictions = best_clf.predict(test_features)\n",
        "\n",
        "# Danh gia hieu suat tot nhat tren mo hinh tap test\n",
        "accuracy = accuracy_score(test_labels, predictions)\n",
        "print(\"Accuracy on test Set\", accuracy)"
      ],
      "metadata": {
        "id": "hTtFxcqlqQWG"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}