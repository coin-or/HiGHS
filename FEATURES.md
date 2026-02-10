## Code changes

Following PR [#2812](https://github.com/ERGO-Code/HiGHS/pull/2812),
HiGHS can read LP files with keywords as constraint names.

Following PR [#2818](https://github.com/ERGO-Code/HiGHS/pull/2818), a
potential data race in the HiGHS multithreading system has been fixed

Following PR [#2825](https://github.com/ERGO-Code/HiGHS/pull/2825),
potential conflict with METIS symbols has been eliminated.

Following PR [#2832](https://github.com/ERGO-Code/HiGHS/pull/2832),
potential conflict with AMD and RCM symbols has been eliminated.

Following PR [#2834](https://github.com/ERGO-Code/HiGHS/pull/2834),
there is some minimal documentation of the `highspy`modelling
language.

Following PR [#2837](https://github.com/ERGO-Code/HiGHS/pull/2837),
the use of the logging callback is independent to the settings of the
`output_flag`, `log_to_console` and `output_flag` options.

## Build changes

