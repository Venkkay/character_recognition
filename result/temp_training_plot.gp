set title 'Training curve'
set xlabel 'Iterations'
set ylabel 'Total error'
set grid
set key top left
plot [1:*] [0:2] '../result/training.dat' using 1:2 with line title 'Training total errors'
pause -1 'Appuyez sur une touche pour continuer...'
