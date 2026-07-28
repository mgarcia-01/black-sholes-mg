cd c_src
gcc -std=c99 -O2 -Wall -Wextra -c mset_rul.c -DMSET_RUL_NO_MAIN -o mset_rul.o
gcc -std=c99 -O2 -Wall -Wextra -c mset_visualizer.c -o mset_visualizer.o
gcc -std=c99 -O2 -Wall -Wextra -c mset_demo_viz.c -o mset_demo_viz.o
gcc mset_rul.o mset_visualizer.o mset_demo_viz.o -lm -o mset_demo_viz
./mset_demo_viz