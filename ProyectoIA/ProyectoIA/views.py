import pandas as pd
import numpy as np   
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
import io, base64
from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
from apyori import apriori
from os import path
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import linear_model, model_selection
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score
matplotlib.use('Agg')
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection


def inicio(request): 
    vista_inicio = loader.get_template('inicio.html')
    documento = vista_inicio.render()
    return HttpResponse(documento)


def aprioriP(request,**kwargs):
    if request.method == "POST": 
        archivo = request.POST.get('archivo')
        if archivo == "":
            archivo = request.POST.get('tmp')
        soporte = request.POST.get('soporte')
        if soporte == "":
            soporte = request.POST.get('soportetmp')
        confianza = request.POST.get('confianza')
        if confianza == "":
            confianza = request.POST.get('confianzatmp')
        elevacion = request.POST.get('elevacion')
        if elevacion == "":
            elevacion = request.POST.get('elevaciontmp')
        info = "Archivo: "+archivo+ " ⇨ Soporte: "+soporte+" ,Confianza: "+confianza+", Elevacion: "+elevacion
        archivoR = path.abspath("ProyectoIA/Documentos/"+archivo)
        DM = pd.read_csv(archivoR,header=None)
        T = pd.DataFrame(DM.values.reshape(-1).tolist())
        T['Frecuencia'] = 0
        T = T.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        T['Porcentaje'] = (T['Frecuencia'] / T['Frecuencia'].sum()) #Porcentaje
        T = T.rename(columns={0 : 'Item'})
        ML = DM.stack().groupby(level=0).apply(list).tolist()
        results = apriori(ML, min_support=float(soporte), min_confidence=float(confianza), min_lift=float(elevacion))
        reglas = list(results)
        LF = []
        NR = []
        if len(reglas) == 0:
            info = info + " ⇨ SIN RESULTADOS"
        else:
            for item in reglas:
                Emparejar = item[0]
                items = [x for x in Emparejar] 
                LF = [str(item[0])[11:-2].replace("'",""),str(item[1]),str(item[2][0][2]),str(item[2][0][3])]
                NR.append(LF)
        fig, ax = plt.subplots(figsize=(16,16))
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        plt.barh(T['Item'],width=T['Frecuencia'],color='red')
        flike = io.BytesIO()
        fig.savefig(flike)
        b64 = base64.b64encode(flike.getvalue()).decode()
        return render(request, "apriori.html",{'b64':b64,'reglas':NR,'archivo':archivo,'soporte':soporte,'confianza':confianza,'elevacion':elevacion,'info':info})
    return render(request, "apriori.html")   

def metricas(request):
    if request.method == "POST":
        archivo = request.POST.get('archivo')
        if archivo == "":
            archivo = request.POST.get('tmp')
        archivoR = path.abspath("ProyectoIA/Documentos/"+archivo)
        tipo = request.POST.get('tipo')
        maximo = len(pd.read_csv(archivoR))
        if tipo == "euclidean":
            tipoNom = "Euclidiana"
        elif tipo == "chebyshev":
            tipoNom = "Chebyshev"
        elif tipo == "cityblock":
            tipoNom = "Manhattan"
        else:
            tipoNom = "Minkowski"
        envioNombre = "Archivo: "+archivo+ " ⇨ Tipo : "+ tipoNom
        return render(request,"buscarMetricas.html",{'archivo':archivo,'nombre':envioNombre, "maximo":maximo, "tipo":tipo})
    return render(request,"metricas.html")

def buscarMetricas(request):
    if request.method == "POST":
        archivo = request.POST.get('tmp')
        archivoR = path.abspath("ProyectoIA/Documentos/"+archivo)
        tipo = request.POST.get('tipotmp')
        ValX = request.POST.get('numeroX')
        ValY = request.POST.get('numeroY')
        tipoRango = request.POST.get('rango')
        DM = pd.read_csv(archivoR)
        if(tipo == "minkowski"):
            Dist = cdist(DM,DM,metric=tipo, p=1.5)
        else:
            Dist = cdist(DM,DM,metric=tipo)
        MDist = pd.DataFrame(Dist)
        if (tipoRango == "UV"):
            val = MDist[int(ValX)][int(ValY)]
            cadena = "Distancia entre los registros "+""+ValX+" y "+ValY+" = "
            loop = "A"
        if (tipoRango == "RV"):
            val = cdist(DM.iloc[int(ValX)-1:int(ValY)], DM.iloc[int(ValX)-1:int(ValY)],metric=tipo)
            cadena = "Distancias entre el rango de registros de "+""+ValX+" a "+ValY+" = "
            loop = [*range(int(ValX),int(ValY)+1)]
            loop = [str(x) for x in loop]
            val = pd.DataFrame(val)
            val.insert(0," ",loop,True)
            loop.insert(0," ")
        if tipo == "euclidean":
            tipoNom = "Euclidiana"
        elif tipo == "chebyshev":
            tipoNom = "Chebyshev"
        elif tipo == "cityblock":
            tipoNom = "Manhattan"
        else:
            tipoNom = "Minkowski"
        envioNombre = "Archivo Usado: "+archivo+ " ⇨ Tipo : "+ tipoNom
        maximo = len(MDist)
        
        
        return render(request,"buscarMetricas.html",{'archivo':archivo,'nombre':envioNombre,"maximo":maximo, "tipo":tipo, "val":np.array(val),"cadena":cadena, "tipoRango": tipoRango, "loop":loop})    
    return render(request,"buscarMetricas.html")


def clustering(request):
    if request.method == "POST":  
        archivo = request.POST.get('archivo')
        if archivo == "":
            archivo = request.POST.get('tmp')
        archivoR = path.abspath("ProyectoIA/Documentos/"+archivo)
        DM = pd.read_csv(archivoR)
        tipo = request.POST.get("tipo")
        variables = list(DM)
        Corr = DM.corr(method='pearson')
        MI =np.triu(Corr)
        fig, ax = plt.subplots(figsize=(15,7))
        sns.heatmap(Corr,cmap='RdBu_r', annot=True, mask=MI)
        flike = io.BytesIO()
        fig.savefig(flike)
        b64 = base64.b64encode(flike.getvalue()).decode()
        if tipo == "J":
            tipoNom = "Jerárquico"
        elif tipo == "P":
            tipoNom = "Particional"
        else:
            tipoNom = "Jerárquico y Particional"
        envioNombre = "Archivo: "+archivo+ " ⇨ Tipo : "+ tipoNom
        return render(request,"buscarClustering.html",{'b64':b64,'variables':variables,'correlacion':np.array(Corr), 'tipo':tipo, 'archivo':archivo, 'nombre':envioNombre, 'mostrar':"hidden"})
    return render(request,"clustering.html")


def buscarClustering(request,**kwargs):
    if request.method == "POST":  
        archivo = request.POST.get('tmp')
        tipo = request.POST.get("tipotmp")
        b64 = request.POST.get("b64tmp")
        seleccion = request.POST.getlist('seleccion')
        numClusteres = request.POST.get("clusteres")
        archivoR = path.abspath("ProyectoIA/Documentos/"+archivo)
        DM = pd.read_csv(archivoR)
        Corr = np.triu(DM.corr(method='pearson'))
        variables = list(DM)
        MP = np.array(DM[seleccion])
       # pd.DataFrame(MP)
        MEstandarizada = StandardScaler().fit_transform(MP)
        val = [*range(int(numClusteres))]
        val = [str(x) for x in val]
        if tipo == "J":
            tipoNom = "Jerárquico"
            MJerarquico = AgglomerativeClustering(n_clusters=int(numClusteres), linkage='complete', affinity='euclidean')
            MJerarquico.fit_predict(MEstandarizada)
            variableRes= set(variables) - set(seleccion)
            nuevoArch = DM.drop(columns=variableRes)
            nuevoArch['cluster'] = MJerarquico.labels_
            clusteres = pd.DataFrame(np.array(nuevoArch.groupby(['cluster'])['cluster'].count()))
            clusteres.insert(0,"cluster",val,True)
            centroides = pd.DataFrame(np.array(nuevoArch.groupby('cluster').mean()))
            centroides.insert(0,"cluster",val,True)

        elif tipo == "P":
            tipoNom = "Particional"
            MParticional = KMeans(n_clusters=int(numClusteres), random_state=0).fit(MEstandarizada)
            MParticional.predict(MEstandarizada)
            variableRes= set(variables) - set(seleccion)
            nuevoArch = DM.drop(columns=variableRes)
            nuevoArch['cluster'] = MParticional.labels_
            clusteres = pd.DataFrame(np.array(nuevoArch.groupby(['cluster'])['cluster'].count()))
            clusteres.insert(0,"cluster",val,True)
            centroides = pd.DataFrame(np.array(nuevoArch.groupby('cluster').mean()))
            centroides.insert(0,"cluster",val,True)
        envioNombre = "Archivo Usado: "+archivo+ " ⇨ Tipo : "+ tipoNom
        return render(request,"buscarClustering.html",{'b64':b64,'dato':np.array(centroides),'cluster':np.array(clusteres),'variables':variables,'correlacion':Corr, 'tipo':tipo, 'seleccion':seleccion, 'archivo':archivo, 'nombre':envioNombre, 'mostrar':" "})
    return render(request,"clustering.html")

def clasificacion(request):
    if request.method == "POST":  
        archivoR = path.abspath("ProyectoIA/Documentos/WDBCOriginal.csv")
        DM = pd.read_csv(archivoR)
        DM = DM.replace({'M':0,'B':1})
        textura = request.POST.get('textura')
        area = request.POST.get('area')
        suavidad = request.POST.get('suavidad')
        compacidad = request.POST.get('compacidad')
        simetria = request.POST.get('simetria')
        dimension = request.POST.get('dimension')
        X = np.array(DM[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']])
        Y = np.array(DM[['Diagnosis']])
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 1234, shuffle = True)
        #pd.DataFrame(X_train)
        #pd.DataFrame(Y_train)
        Clasificacion = linear_model.LogisticRegression()
        Clasificacion.fit(X_train, Y_train)
        Probabilidad = Clasificacion.predict_proba(X_validation)
        Predicciones = Clasificacion.predict(X_validation)
        porcentaje = Clasificacion.score(X_validation,Y_validation)
        Y_Clasificacion = Clasificacion.predict(X_validation)
        arreglo = pd.DataFrame({'Texture': [float(textura)], 'Area': [float(area)], 'Smoothness': [float(suavidad)], 'Compactness': [float(compacidad)], 'Symmetry': [float(simetria)],'FractalDimension': [float(dimension)]})
        res = Clasificacion.predict(arreglo)

        porcentaje = "Confianza del "+str(round(porcentaje*100,2)) + "%"
        if res == [1]:
            res = "BENIGNO"
            color = "#62bd4b"
        else:
            res = "MALIGNO"
            color = "#bd4b4b"
        return render(request,"clasificacion.html",{'porcentaje':np.array(porcentaje), 'resultado':np.array(res),'color':color, 'textura':textura, 'area':area,'suavidad':suavidad,'compacidad':compacidad,'simetria':simetria,'dimension':dimension})
    return render(request,"clasificacion.html")

def arboles(request):
    if request.method == "POST":  
        archivo = request.POST.get('archivo')
        if archivo == "":
            archivo = request.POST.get('tmp')
        archivoR = path.abspath("ProyectoIA/Documentos/"+archivo)
        DM = pd.read_csv(archivoR)
        tipo = request.POST.get("tipo")
        variables = list(DM)
        if tipo == "P":
            tipoNom = "Pronóstico"
        elif tipo == "C":
            tipoNom = "Clasificación"
        envioNombre = "Archivo: "+archivo+ " ⇨ Tipo : "+ tipoNom
        return render(request,"buscarArboles.html",{'variables':variables, 'tipo':tipo, 'archivo':archivo, 'nombre':envioNombre, 'mostrar':"hidden"})
    return render(request,"arboles.html")

def buscarArboles(request):
    if request.method == "POST":  
        archivo = request.POST.get('tmp')
        tipo = request.POST.get("tipotmp")
        seleccion = request.POST.getlist('seleccion') 
        clase = request.POST.get('clase')
        
        if seleccion == []:
            seleccion = request.POST.get('selecciontmp')
            if seleccion.find("'"):
                seleccion = seleccion.replace(" ","").replace("'","").replace("[",'').replace("]",'')
                seleccion = seleccion.split(",")
        if clase == None:
            clase = request.POST.get('clasetmp')
        varP = "Variables Predictoras : "
        varC = "Variable Clase : "+ clase
        for x in seleccion:
            varP = varP+str(x)+", " 
        archivoR = path.abspath("ProyectoIA/Documentos/"+archivo)
        DM = pd.read_csv(archivoR)
        variables = list(DM)     
        if tipo == "P":
            tipoNom = "Pronóstico"
        elif tipo == "C":
            tipoNom = "Clasificación"
        envioNombre = "Archivo: "+archivo+ " ⇨ Tipo : "+ tipoNom 
        return render(request,"resArboles.html",{'seleccion':seleccion,'varP':varP[:-2],'varC':varC,'clase':clase,'variables':variables, 'tipo':tipo, 'archivo':archivo,'nombre':envioNombre, 'mostrar':" "})
    return render(request,"buscarArboles.html",)



def resArboles(request,**kwargs):
    if request.method == "POST":  
        archivo = request.POST.get('tmp')
        tipo = request.POST.get("tipotmp")
        seleccion = request.POST.get('selecciontmp') 
        clase = request.POST.get('clasetmp')
        varP = request.POST.get('varPtmp')
        varC = request.POST.get('varCtmp')
        if seleccion.find("'"):
            seleccion = seleccion.replace(" ","").replace("'","").replace("[","").replace("]","")
            seleccion = seleccion.split(",")
        archivoR = path.abspath("ProyectoIA/Documentos/"+archivo)
        DM = pd.read_csv(archivoR)
        DM = DM.replace({'M':0,'B':1})
        a = seleccion
        variables = list(DM)
        X = np.array(DM[seleccion])
        Y = np.array(DM[clase])
        X_train,X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.2, random_state=1234, shuffle=True)       
        if tipo == "P":
            tipoNom = "Pronóstico"
            AD = DecisionTreeRegressor()
            AD.fit(X_train,Y_train)
            Y_Pronostico = AD.predict(X_test)
            porcentaje = r2_score(Y_test,Y_Pronostico)
            porcentaje = "Confianza del "+str(round(porcentaje*100,2)) + "%"
            fig, ax = plt.subplots(figsize=(16,16))
            plot_tree(AD,feature_names=seleccion)
            flike = io.BytesIO()
            fig.savefig(flike)
            b64 = base64.b64encode(flike.getvalue()).decode()      

        elif tipo == "C":
            tipoNom = "Clasificación" 
            AD = DecisionTreeClassifier(random_state=0)
            #AD = DecisionTreeRegressor()
            AD.fit(X_train,Y_train)
            porcentaje = AD.score(X_test,Y_test)
            porcentaje = "Confianza del "+str(round(porcentaje*100,2)) + "%"
            Y_clasificacion = AD.predict(X_test)
            fig, ax = plt.subplots(figsize=(10,10))
            plot_tree(AD,feature_names=seleccion, class_names=Y_clasificacion)
            flike = io.BytesIO()
            fig.savefig(flike)
            b64 = base64.b64encode(flike.getvalue()).decode()          
        envioNombre = "Archivo Usado: "+archivo+ " ⇨ Tipo : "+ tipoNom
        objConsulta = ""
        for x in seleccion:
            objConsulta = objConsulta+ x + ":[" + int(request.POST.get(x))+"],"
        objConsulta = objConsulta[:-1]
        A = pd.DataFrame({objConsulta})        
        B = AD.predict(A)
        

        return render(request,"resArboles.html",{'OBJ':objConsulta,'b64':b64,'porcentaje':porcentaje,'varP':varP,'varC':varC,'seleccion':seleccion,'clase':clase,'variables':variables, 'tipo':tipo, 'seleccion':seleccion, 'archivo':archivo, 'nombre':envioNombre, 'mostrar':" "})
    return render(request,"arboles.html")
