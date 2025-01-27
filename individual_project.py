```python
import numpy as np 

import matplotlib.pyplot as plt 

from PIL import Image 

  

# Wczytaj plik TIFF (RGB) 

image = Image.open('76740_1132577_M-34-45-A-b-4-4.tif') 

   

# Wyświetlenie obrazu 

plt.imshow(image) 

plt.axis('off')  # Ukrywa osie 

plt.show()  
```


    
![png](output_0_0.png)
    



```python
# Konwersja obrazu na tablicę numpy (RGB) 

rgb_array = np.array(image) 

  

# Rozdzielenie pasm RGB 

red_image = rgb_array[:, :, 0]  # Czerwony kanał 

green_image = rgb_array[:, :, 1]  # Zielony kanał 

blue_image = rgb_array[:, :, 2]  # Niebieski kanał 

 

# Wyświetlenie rozmiaru obrazu 

print(f"Size of the image: {rgb_array.shape}") 
```

    Size of the image: (9648, 9187, 3)
    


```python
# Zakładamy, że kanał czerwony to pierwszy, a NIR to drugi 

red = rgb_array[:, :, 0]  # Kanał czerwony 

nir = rgb_array[:, :, 1]  # Kanał bliskiej podczerwieni (NIR) 

  

# Obliczanie NDVI z zabezpieczeniem przed dzieleniem przez zero 

epsilon = 1e-6  # Mała wartość, aby uniknąć zerowego mianownika 

ndvi = (nir - red) / (nir + red + epsilon) 

  

# Zamiana wartości NaN i nieskończonych na 0 

ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0) 

  

# Normalizacja do zakresu [0, 255] dla wizualizacji 

ndvi_normalized = ((ndvi + 1) / 2) * 255 

  

# Zamiana wartości spoza zakresu [0, 255] na 0 (dla pewności) 

ndvi_normalized = np.clip(ndvi_normalized, 0, 255) 

  

# Konwersja na obraz do wizualizacji 

ndvi_image = Image.fromarray(ndvi_normalized.astype(np.uint8)) 

  

# Wyświetlenie obrazu NDVI 

plt.imshow(ndvi_normalized, cmap="RdYlGn") 

plt.colorbar(label="NDVI") 

plt.title("NDVI Visualization") 

plt.show() 
```


    
![png](output_2_0.png)
    



```python
# Funkcja do obliczania SAVI
def calculate_savi(red, green, L=0.5):
    savi = (green - red) * (1 + L) / (green + red + L)
    return savi

# Obliczanie SAVI
savi = calculate_savi(red_image, green_image)

# Normalizacja i wizualizacja dla SAVI
savi_normalized = np.clip((savi + 1) * 255 / 2, 0, 255)  # Normalizacja do zakresu 0-255
plt.imshow(savi_normalized, cmap='YlGn')  # Wizualizacja z mapą kolorów
plt.colorbar()
plt.title("SAVI Visualization")
plt.show()
```


    
![png](output_3_0.png)
    



```python
# Funkcja do obliczania GCI (zabezpieczenie przed dzieleniem przez zero)
def calculate_gci(red, green, epsilon=1e-6):
    # Dodaj epsilon, aby uniknąć dzielenia przez zero
    gci = green / (red + epsilon) - 1
    return gci

# Obliczanie GCI
gci = calculate_gci(red_image, green_image)

# Normalizacja i wizualizacja dla GCI
gci_normalized = np.clip((gci + 1) * 255 / 2, 0, 255)  # Normalizacja do zakresu 0-255
plt.imshow(gci_normalized, cmap='YlGn')  # Wizualizacja z mapą kolorów
plt.colorbar()
plt.title("GCI Visualization")
plt.show()
```


    
![png](output_4_0.png)
    



```python
# Klasyfikacja na podstawie wartości NDVI 

classified = np.zeros_like(ndvi, dtype=np.uint8) 

  

# Wysoka roślinność (NDVI > 0.5) 

classified[ndvi > 0.5] = 3 

# Umiarkowana roślinność (0.2 < NDVI <= 0.5) 

classified[(ndvi > 0.2) & (ndvi <= 0.5)] = 2 

# Brak roślinności (NDVI <= 0.2) 

classified[ndvi <= 0.2] = 1 

  

# Wyświetlenie obrazu klasyfikacji 

plt.imshow(classified, cmap="tab10") 

plt.title("Classified Image") 

plt.show() 

```


    
![png](output_5_0.png)
    



```python
# Statystyka klas 

unique, counts = np.unique(classified, return_counts=True) 

class_distribution = dict(zip(unique, counts)) 

  

# Obliczenie procentowego udziału 

total_pixels = classified.size 

class_percentages = {k: (v / total_pixels) * 100 for k, v in class_distribution.items()} 

  

# Wyświetlenie wyników 

print("Rozkład pikseli w klasach:") 

for cls, count in class_distribution.items(): 

    print(f"Klasa {cls}: {count} pikseli ({class_percentages[cls]:.2f}%)") 
```

    Rozkład pikseli w klasach:
    Klasa 1: 55127117 pikseli (62.19%)
    Klasa 2: 2312278 pikseli (2.61%)
    Klasa 3: 31196781 pikseli (35.20%)
    


```python
import matplotlib.pyplot as plt 

from matplotlib.ticker import FuncFormatter 

  

# Funkcja formatująca liczby na osi Y 

def format_y(value, _): 

    if value >= 1e6: 

        return f"{value / 1e6:.1f}M"  # Miliony 

    elif value >= 1e3: 

        return f"{value / 1e3:.1f}k"  # Tysiące 

    return f"{int(value)}"  # Jednostki 

  

# Wizualizacja rozkładu klas 

classes = list(class_distribution.keys()) 

counts = list(class_distribution.values()) 

  

plt.bar(classes, counts, color=["red", "green", "blue"], tick_label=["Brak roślinności", "Umiarkowana roślinność", "Wysoka roślinność"]) 

plt.xlabel("Klasy") 

plt.ylabel("Liczba pikseli") 

plt.title("Rozkład klas w obrazie") 

  

# Formatowanie osi Y 

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y)) 

plt.show() 
```


    
![png](output_7_0.png)
    



```python
# Powierzchnia każdej klasy (opcjonalnie) 

# Przyjmujemy, że rozdzielczość przestrzenna obrazu to 10x10 metrów na piksel 

pixel_area_m2 = 10 * 10  # Powierzchnia jednego piksela w metrach kwadratowych 

pixel_area_ha = pixel_area_m2 / 10000  # Powierzchnia w hektarach 

  

class_areas = {k: v * pixel_area_ha for k, v in class_distribution.items()} 

print("\nPowierzchnia klas (w hektarach):") 

for cls, area in class_areas.items(): 

    print(f"Klasa {cls}: {area:.2f} ha") 
```

    
    Powierzchnia klas (w hektarach):
    Klasa 1: 551271.17 ha
    Klasa 2: 23122.78 ha
    Klasa 3: 311967.81 ha
    


```python

```
