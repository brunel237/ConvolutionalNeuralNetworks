def selection(tab):
    taille = len(tab)
    for i in range(taille):
        pivot = i
        for j in range(i+1, taille):
            if tab[j]<tab[pivot]:
                pivot = j
                tab[i], tab[pivot] = tab[pivot], tab[i]
    return tab

def insertion(tab):
    taille = len(tab)
    for i in range(1,taille):
        cle = tab[i]
        j= i-1
        while j>=0 and tab[j]>cle:
            tab[j+1]=tab[j]
            j = j-1
        tab[j+1] = cle
    return tab
        
print(insertion([5,3,1,0,6,7,10,8]))


# def convolution(tableau_images, tableau_filtres):
#     nombre_images = len(tableau_images)
#     taille_image = len(tableau_images[0])
#     nombre_filtres = len(tableau_filtres)
#     taille_filtres = len(tableau_filtres[0])
#     taille_resultat = taille_image-taille_filtres +1

#     tableau_final = []

#     for i in range(nombre_images):
#         for j in range(nombre_filtres):
#             resultat = [[0]*taille_resultat for i in range(taille_resultat)]
#             for x in range(taille_resultat):
#                 for y in range(taille_resultat):
#                     som = 0
#                     for m in range(taille_filtres):
#                         for n in range(taille_filtres):
#                             som += tableau_filtres[j][m][n]
#                     resultat[x][y] = som
#             tableau_final.append(resultat)

#     return tableau_final


















