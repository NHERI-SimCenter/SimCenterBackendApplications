
// dummy_driver.c
//    - simple c application to create a results.out file with a singular 0.0 entry
// written: fmk 04/26

#include <stdio.h>
int main() {
    FILE *f = fopen("results.out", "w");
    fprintf(f, "0.0");
    fclose(f);
    return 0;
}
