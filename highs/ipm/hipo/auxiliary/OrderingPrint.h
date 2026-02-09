#ifndef HIPO_ORDERING_PRINT_H
#define HIPO_ORDERING_PRINT_H

#ifndef NDEBUG
#define HIGHS_ORDERING_PRINT(params) printf params
#else
#define HIGHS_ORDERING_PRINT(params)
#endif

#endif