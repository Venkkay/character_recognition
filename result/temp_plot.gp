set title 'Generalisation curves'
set xlabel 'Noise percentage'
set ylabel 'Error rate'
set grid
set key top left
plot '../result/generalisation.dat' using 1:2 with lines title 'Pattern 0', '../result/generalisation.dat' using 1:3 with lines title 'Pattern 1', '../result/generalisation.dat' using 1:4 with lines title 'Pattern 2', '../result/generalisation.dat' using 1:5 with lines title 'Pattern 3', '../result/generalisation.dat' using 1:6 with lines title 'Pattern 4', '../result/generalisation.dat' using 1:7 with lines title 'Pattern 5', '../result/generalisation.dat' using 1:8 with lines title 'Pattern 6', '../result/generalisation.dat' using 1:9 with lines title 'Pattern 7', '../result/generalisation.dat' using 1:10 with lines title 'Pattern 8', '../result/generalisation.dat' using 1:11 with lines title 'Pattern 9'
