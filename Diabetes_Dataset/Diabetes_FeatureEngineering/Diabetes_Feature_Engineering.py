# Feature Engineering

# İş Problemi:
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
# edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
# geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
# gerçekleştirmeniz beklenmektedir.

# Veri Seti Hikayesi:
# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
# Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
# yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# DEĞİŞKENLER:
# Pregnancies = Hamilelik sayısı
# Glucose Oral =  glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
# Blood Pressure  = Kan Basıncı (Küçük tansiyon) (mm Hg)
# SkinThickness =  Cilt Kalınlığı
# Insulin  = 2 saatlik serum insülini (mu U/ml)
# DiabetesPedigreeFunction =  Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# BMI =  Vücut kitle endeksi
# Age =  Yaş (yıl)
# Outcome  = Hastalığa sahip (1) ya da değil (0)


# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
           # Adım 1: Genel resmi inceleyiniz.
           # Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
           # Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
           # Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
           # Adım 5: Aykırı gözlem analizi yapınız.
           # Adım 6: Eksik gözlem analizi yapınız.
           # Adım 7: Korelasyon analizi yapınız.

# GÖREV 2: FEATURE ENGINEERING
           # Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
           # değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri
           # 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere
           # işlemleri uygulayabilirsiniz.
           # Adım 2: Yeni değişkenler oluşturunuz.
           # Adım 3:  Encoding işlemlerini gerçekleştiriniz.
           # Adım 4: Numerik değişkenler için standartlaştırma yapınız.
           # Adım 5: Model oluşturunuz.


# Gerekli Kütüphane ve Fonksiyonlar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
# from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("projeler/6)Diabets(Feature_Engineering)/diabetes/diabetes.csv")
df = df_.copy()
df.shape
##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("\n//////Shape//////")
    print(f'Shape     : {dataframe.shape}\n'
          f'Size      : {dataframe.size}\n'
          f'Dimension : {dataframe.ndim}')
    print("\n//////Types//////")
    print(dataframe.dtypes)
    print("\n//////Head//////")
    print(dataframe.head(head))
    print("\n//////Tail//////")
    print(dataframe.tail(head))
    print("\n//////Random Sampling//////")
    print(dataframe.sample(head))
    print("\n//////Missing Values//////")
    print(dataframe.isnull().sum())
    print("\n//////Duplicated Values//////")
    print(dataframe.duplicated().sum())
    print("\n//////Unique Values//////")
    print(dataframe.nunique())
    print("\n//////Describe//////")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# NOT: Size --> toplam hücre (eleman) sayısını döndürür. DataFrame'in satır sayısı ile sütun sayısının çarpımına eşittir.
# NOT: Dimension -->  DataFrame'in boyutunu (dimension) döndürür. Eğer DataFrame bir dizi (1D) ise ndim değeri 1 olacaktır.
# Eğer DataFrame bir tablo (2D) ise ndim değeri 2 olacaktır.
check_df(df)

# Glucose degeri sıfır olabilir mi?
# Insulin degeri sıfır olabilir mi ?
# Kan basıncı sıfır olabilir mi?
# Veri setinde eksik degereler vardı da sıfır basıldı?
# Insulin degerinde 95 ceyreklikten max degere buyuk bir sıcrayıs var bu da aykırı deger olabileceginin bir sinyali.


##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################
#///////////////////////////////////////////////////

# Fonksiyonumuz: Vahit hocadan farklı olarak Pregnancyleri de kategorik değişkene çevirmek için cat_th = 18 yaptım.
def grab_col_names(dataframe, cat_th=18, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

### NOT : Kategorik değişken dediğimiz zaman akla value countslarını almak gelecek, numerik dediğimiz zaman ise describe yani oranlar akla gelecek.

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################
# amacım değişkene dair degerlerin oranına göz atmak.
def cat_summary(dataframe, col_name, plot=False): # plot:true olursa if çalışır.
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),  #değişkende hangi degerden kacar adet var?
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})) # deger adetlerini toplam deger sayısına bölümü oran verir.
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# kategorik değişkenimde deniyorum.
cat_summary(df, "Outcome", True)
cat_summary(df, "Pregnancies", True)


for col in cat_cols:
    cat_summary(df, col)


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):  # plot:true olursa if çalışır.
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] # hangi ceyreklikleri istiyorum?
    print(dataframe[numerical_col].describe(quantiles).T) # istedigim ceyreklikler bazında describe göz atıyorum.

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)     # block=True demeyince saçma bir grafik veriyor.


for col in num_cols: # num_cols: grab_col_names fonksiyonundan elde ettigim numerik değişkenlerim.
    num_summary(df, col, plot=True)


##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

# numerik degişkenlerin target değişkene göre ortalamalarını inceleyelim:
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


"""
         Glucose
Outcome         
0        109.980
1        141.257

         Insulin
Outcome         
0         68.792
1        100.336

         SkinThickness
Outcome               
0               19.664
1               22.164
"""
# Değerlere baktığımız zaman diyabet hastası olanların bu değerlerininde yüksek olduğunu görüyoruz.


##################################
# KORELASYON
##################################

# Korelasyon, olasılık kuramı ve istatistikte iki rassal değişken arasındaki doğrusal ilişkinin yönünü ve gücünü belirtir

df.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Korelasyona baktığımızda yüksek korelasyonlu değerlerin pek olmadığını görüyoruz.


##################################
# BASE MODEL KURULUMU
##################################

# Amacımız herhangi bir işlem yapmadan başarımız ne durumda?
# Sonrasıyla karsılastıralım.

y = df["Outcome"] # Bağımlı ve bağımsız değişkeni seçmemiz gerekiyor. Bu çalışmadaki bağımlı değişken Outcome değişkeni.
X = df.drop("Outcome", axis=1) # Bu çalışmadaki bağımsız değişken Outcome  dışındaki değişkenler.
# Veri setini train ve test olarak 2'ye ayırıyoruz. Train seti üzerinde model kurucaz, test seti ile kurduğumuz bu modeli test ediyor olucaz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
# test_size=0.30 --> Yani 100 tane veri varsa bunların 30 tanesini test için 70 tanesini modeli eğitmek için kullanıyoruz.

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train) #1 satırda model kuruldu. Makine öğrenmesi yaptık.
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}") # basarı oranı
print(f"Recall: {round(recall_score(y_pred,y_test),3)}") # Gercekte diyabet olanların kacına diyabet dedigi
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}") # Recall'in tam tersi. Model tarafından tahmin edilen degerlerin kac tanesi diyabet
print(f"F1: {round(f1_score(y_pred,y_test), 2)}") # Recall ve precision ortalaması
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}") # farklı sınıflandırma esik degerlerine göre basarı

# Accuracy: 0.77
# Recall: 0.706 # pozitif sınıfın ne kadar başarılı tahmin edildiği
# Precision: 0.59 # Pozitif sınıf olarak tahmin edilen değerlerin başarısı
# F1: 0.64
# Auc: 0.75


# Model hangi değişkene daha cok önem varmiş?
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)



#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

df.isnull().sum() # eksik degerler yoktu. Fakat sıfır olamayacak değişkenlere sıfır atanmıstı.
df.describe()

# Bir insanda Pregnancies ve Outcome dışındaki değişken değerleri 0 olamayacağı bilinmektedir.
# Bundan dolayı bu değerlerle ilgili aksiyon kararı alınmalıdır. 0 olan değerlere NaN atanabilir .


# 1. YOL :
zero_value = [ "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for z in zero_value:
    print(z, df.loc[df[z] == 0].shape)
    df[z] = np.where(df[z] == 0, np.nan, df[z])
# Bu loopta 0 değerine sahip olanları NaN yapıyor. Bunu fonka da çevirebilirsin. Önemli bir şey.


# 2. YOL :
# minimum degeri sıfır olamayacak değişkenler yakalanıyor.
# kategorik değişkenler hariç bırakılıyor.
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns

# where ile eger ki şart saglanıyorsa NAN yazacagım, saglanmıyorsa oldugu gibi yazacagım.
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

# Gözlem birimlerinde 0 olan degiskenlerin her birisine gidip 0 iceren gozlem degerlerini NaN ile değiştirdik.

# Eksik Gözlem Analizi
df.isnull().sum()          # Ve şimdi eksik değerler geldi


# artık eksik degerleri (NAN) inceleyebiliriz.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] # eksik deger varsa na_columns değişkeninde tutulur.
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False) # sıralamanın sebebi ilk olarak fazla eksik degere sahip eğişkenleri görmek istememiz.
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False) # eksik degerlerin tüm değerler içerisindeki oranı
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio']) # kac deger var ve oranını birleştiriyoruz.
    print(missing_df, end="\n")
    if na_name: #na_name true ise degeri döndürür.
        return na_columns

na_columns = missing_values_table(df, na_name=True)


# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
# amacımız eksik degerler ile var olan degerlerin karsılastırmasını yapmak olacak.

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns: # eksik degeri olan değişkenlerde geziyoruz.
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0) # temp_df[col].isnull() degeri true false olarak döndürür. true ise 1: false ise 0 yazar.
        # bu işlemin amacı eksik olan degerlerde var olan degerleri ayrıstırmaktır.
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns # NA barındıran değişkenlerde gezmek istiyorum. yeni değişkene atadım.
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", na_columns)
# Eksikse 1 değilse 0.

"""
                 TARGET_MEAN  Count
Glucose_NA_FLAG                    
0                      0.349    763
1                      0.400      5
"""
# Glukoz değeri içinde eksik değere sahip olanlara 1 flagı atandı. Eksik değeri olanların Targetinin ortalaması %40 ve 5 tane eksik değere sahip.
# Eksik değere sahip olmayanların Targetinin ortalamsı ise %34 ve 763 tane değeri varmış.



# Eksik Değerlerin Doldurulması
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()


df.isnull().sum()

df.describe().T


##################################
# AYKIRI DEĞER ANALİZİ
##################################


# aykırı degerler için limit belirleme
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# aykırı deger var mı yok mu?
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# aykırı degerleri baskılama
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Aykırı Değer Analizi ve Baskılama İşlemi
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


for col in df.columns:
    print(col, check_outlier(df, col))

df.describe().T

##################################
# ÖZELLİK ÇIKARIMI
##################################
# Burada yapacak olduğumuz işlem 1-0 şeklinde var olan değişkenler üzerinden yeni değişkenler türetmek. Ama bu label, binary encoding değil.
# Var olan değişkenler içinden yeni bir şeyler türetmekle ilgileniyoruz var olanı değiştirmekle değil.

df.nunique()
# Pregnancy, BMI, BloodPressure, Glukoz, Insulin  ve Age üstünden bir ayırma yapabiliriz gibi.

df.head(60)

## 1) AGE ##
# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

## 2) BMI ##
# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
df["NEW_BMI"] = pd.cut(x=df['BMI'], bins= [0, 18.5, 24.9, 29.9, 100], labels = ["Underweight", "Healthy", "Overweight", "Obese"])

## 3) BloodPressure ##
df["NEW_BLOODPRESSURE"] = pd.cut(x = df['BloodPressure'], bins = [0, 80, 110, df["BloodPressure"].max()], labels = ["low", "Ideal", "high"])

## 4) GLUKOZ ##
df["NEW_GLUXOSE"] = pd.cut(x = df["Glucose"], bins = [0, 140, 200, 300], labels = ["Normal", "Prediabetes", "Diabetes"])

## 5) Pregnancy ##
df["NEW_PREGNANCY"] = pd.cut(x = df["Pregnancies"], bins = [0, 3, 6, 12, 20], labels = ["Few-Medium","Many","Too Many", "x2Too Many" ])



# # Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma 3 kırılım yakalandı
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"


# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


# Yaş ve BloodPressure değerlerini bir arada düşünerek kategorik değişken oluşturma
df.loc[(df["NEW_AGE_CAT"] == "mature") & ((df["BloodPressure"] > 0) & (df["BloodPressure"] <= 80)), "NEW_AGE_BLOODPRESSURE_NOM"] = "lowbloodmature"
df.loc[(df["NEW_AGE_CAT"] == "senior") & ((df["BloodPressure"] > 0) & (df["BloodPressure"] <= 80)), "NEW_AGE_BLOODPRESSURE_NOM"] = "lowbloodsenior"
df.loc[(df["NEW_AGE_CAT"] == "mature") & ((df["BloodPressure"] > 80) & (df["BloodPressure"] <= 110)), "NEW_AGE_BLOODPRESSURE_NOM"] = "Idealbloodmature"
df.loc[(df["NEW_AGE_CAT"] == "senior") & ((df["BloodPressure"] > 80) & (df["BloodPressure"] <= 110)), "NEW_AGE_BLOODPRESSURE_NOM"] = "Idealbloodsenior"
df.loc[(df["NEW_AGE_CAT"] == "mature") & ((df["BloodPressure"] > 110)), "NEW_AGE_BLOODPRESSURE_NOM"] = "highbloodmature"
df.loc[(df["NEW_AGE_CAT"] == "senior") & ((df["BloodPressure"] > 110)), "NEW_AGE_BLOODPRESSURE_NOM"] = "highbloodsenior"

# BMI ve Glukoz değerlerini bir arada düşünerek kategorik değişken oluşturma

df.loc[(df["NEW_GLUXOSE"] == "Normal" ) & (df["NEW_BMI"] == "Underweight"), "NEW_BMI_GLUXOSE"] = "NormalUnderweight"
df.loc[(df["NEW_GLUXOSE"] == "Normal" ) & (df["NEW_BMI"] == "Healthy"), "NEW_BMI_GLUXOSE"] = "NormalHealthy"
df.loc[(df["NEW_GLUXOSE"] == "Normal" ) & (df["NEW_BMI"] == "Overweight"), "NEW_BMI_GLUXOSE"] = "NormalOverweight"
df.loc[(df["NEW_GLUXOSE"] == "Normal" ) & (df["NEW_BMI"] == "Obese"), "NEW_BMI_GLUXOSE"] = "NormalObese"
df.loc[(df["NEW_GLUXOSE"] == "Prediabetes" ) & (df["NEW_BMI"] == "Underweight"), "NEW_BMI_GLUXOSE"] = "PrediabetesUnderweight"
df.loc[(df["NEW_GLUXOSE"] == "Prediabetes" ) & (df["NEW_BMI"] == "Healthy"), "NEW_BMI_GLUXOSE"] = "PrediabetesHealthy"
df.loc[(df["NEW_GLUXOSE"] == "Prediabetes" ) & (df["NEW_BMI"] == "Overweight"), "NEW_BMI_GLUXOSE"] = "PrediabetesOverweight"
df.loc[(df["NEW_GLUXOSE"] == "Prediabetes" ) & (df["NEW_BMI"] == "Obese"), "NEW_BMI_GLUXOSE"] = "PrediabetesObese"
df.loc[(df["NEW_GLUXOSE"] == "Diabetes" ) & (df["NEW_BMI"] == "Underweight"), "NEW_BMI_GLUXOSE"] = "DiabetesUnderweight"
df.loc[(df["NEW_GLUXOSE"] == "Diabetes" ) & (df["NEW_BMI"] == "Healthy"), "NEW_BMI_GLUXOSE"] = "DiabetesHealthy"
df.loc[(df["NEW_GLUXOSE"] == "Diabetes" ) & (df["NEW_BMI"] == "Overweight"), "NEW_BMI_GLUXOSE"] = "DiabetesOverweight"
df.loc[(df["NEW_GLUXOSE"] == "Diabetes" ) & (df["NEW_BMI"] == "Obese"), "NEW_BMI_GLUXOSE"] = "DiabetesObese"

df.head()

df.corr()

## 6) INSULIN ##
# İnsulin Değeri ile Kategorik değişken türetmek

def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df.head()

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]









#///////////////////////////////////////////////////
# Adım 2: Yeni değişkenler oluşturunuz.


#///////////////////////////////////////////////////
# Adım 3: Encoding işlemlerini gerçekleştiriniz.


#///////////////////////////////////////////////////
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.


#///////////////////////////////////////////////////
# Adım 5: Model oluşturunuz.