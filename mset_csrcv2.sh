cd c_srcv2

gcc -std=c99 -O2 -Wall -Wextra -c mset_rul.c -DMSET_RUL_NO_MAIN -o mset_rul.o
gcc -std=c99 -O2 -Wall -Wextra -c mset_visualizer.c -o mset_visualizer.o
gcc -std=c99 -O2 -Wall -Wextra -c csv_io.c -o csv_io.o
gcc -std=c99 -O2 -Wall -Wextra -c mset_demo_csv.c -o mset_demo_csv.o
gcc mset_rul.o mset_visualizer.o csv_io.o mset_demo_csv.o -lm -o mset_demo_csv
./mset_demo_csv training.csv test.csv out.html