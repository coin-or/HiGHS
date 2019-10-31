/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2019 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsOptions.cpp
 * @brief
 * @author Julian Hall, Ivet Galabova, Qi Huangfu and Michael Feldmeier
 */
#include "lp_data/HighsOptions.h"
//#include "io/Filereader.h"
//#include "io/HighsIO.h"

inline const char* bool2string(bool b) { return b ? "true" : "false"; }

std::string optionEntryType2string(const HighsOptionType type) {
  if (type == HighsOptionType::BOOL) {
    return "bool";
  } else if (type == HighsOptionType::INT) {
    return "int";
  } else if (type == HighsOptionType::DOUBLE) {
    return "double";
  } else
    return "string";
}

bool commandLineOffChooseOnOk(const string& value) {
  if (value == off_string || value == choose_string || value == on_string) return true;
  HighsLogMessage(HighsMessageType::ERROR, "Command line option value \"%s\" is not one of \"%s\", \"%s\" or \"%s\"\n",
		  value.c_str(), off_string.c_str(), choose_string.c_str(), on_string.c_str());
  return false;
}

bool commandLineSolverOk(const string& value) {
  if (value == simplex_string || value == choose_string || value == ipm_string) return true;
  HighsLogMessage(HighsMessageType::ERROR, "Command line option value \"%s\" is not one of \"%s\", \"%s\" or \"%s\"\n",
		  value.c_str(), simplex_string.c_str(), choose_string.c_str(), ipm_string.c_str());
  return false;
}

bool boolFromString(const std::string value, bool& bool_value) {
  if (
      value == "t" ||
      value == "true" ||
      value == "T" ||
      value == "True" ||
      value == "TRUE") {
    bool_value = true;
  } else if (
      value == "f" ||
      value == "false" ||
      value == "F" ||
      value == "False" ||
      value == "FALSE") {
    bool_value = false;
  } else {
    return false;
  }
  return true;
}

OptionStatus getOptionIndex(const std::string& name, const std::vector<OptionRecord*>& option_records, int& index) {
  int num_options = option_records.size();
  for (index = 0; index < num_options; index++) if (option_records[index]->name == name) return OptionStatus::OK;
  HighsLogMessage(HighsMessageType::ERROR, "getOptionIndex: Option \"%s\" is unknown", name.c_str());
  return OptionStatus::UNKNOWN_OPTION;
}


OptionStatus checkOptions(const std::vector<OptionRecord*>& option_records) {
  bool error_found = false;
  int num_options = option_records.size();
  for (int index = 0; index < num_options; index++) {
    std::string name = option_records[index]->name;
    HighsOptionType type = option_records[index]->type;
    // Check that there are no other options with the same name
    for (int check_index = 0; check_index < num_options; check_index++) {
      if (check_index == index) continue;
      std::string check_name = option_records[check_index]->name;
      if (check_name == name) {
	HighsLogMessage(HighsMessageType::ERROR,
			"checkOptions: Option %d (\"%s\") has the same name as option %d \"%s\"",
			index, name.c_str(),
			check_index, check_name.c_str());
	error_found = true;
      }
    }
    if (type == HighsOptionType::BOOL) {
      // Check bool option
      OptionRecordBool& option = ((OptionRecordBool*)option_records[index])[0];
      // Check that there are no other options with the same value pointers
      bool* value_pointer = option.value;
      for (int check_index = 0; check_index < num_options; check_index++) {
	if (check_index == index) continue;
	OptionRecordBool& check_option = ((OptionRecordBool*)option_records[check_index])[0];
	if (check_option.type == HighsOptionType::BOOL) {
	  if (check_option.value == value_pointer) {
	    HighsLogMessage(HighsMessageType::ERROR,
			    "checkOptions: Option %d (\"%s\") has the same value pointer as option %d (\"%s\")",
			    index, option.name.c_str(),
			    check_index, check_option.name.c_str());
	    error_found = true;
	  }
	}
      }
    } else if (type == HighsOptionType::INT) {
      // Check int option
      OptionRecordInt& option = ((OptionRecordInt*)option_records[index])[0];
      if (checkOption(option) != OptionStatus::OK) error_found = true;
      // Check that there are no other options with the same value pointers
      int* value_pointer = option.value;
      for (int check_index = 0; check_index < num_options; check_index++) {
	if (check_index == index) continue;
	OptionRecordInt& check_option = ((OptionRecordInt*)option_records[check_index])[0];
	if (check_option.type == HighsOptionType::INT) {
	  if (check_option.value == value_pointer) {
	    HighsLogMessage(HighsMessageType::ERROR,
			    "checkOptions: Option %d (\"%s\") has the same value pointer as option %d (\"%s\")",
			    index, option.name.c_str(),
			    check_index, check_option.name.c_str());
	    error_found = true;
	  }
	}
      }
    } else if (type == HighsOptionType::DOUBLE) {
      // Check double option
      OptionRecordDouble& option = ((OptionRecordDouble*)option_records[index])[0];
      if (checkOption(option) != OptionStatus::OK) error_found = true;
      // Check that there are no other options with the same value pointers
      double* value_pointer = option.value;
      for (int check_index = 0; check_index < num_options; check_index++) {
	if (check_index == index) continue;
	OptionRecordDouble& check_option = ((OptionRecordDouble*)option_records[check_index])[0];
	if (check_option.type == HighsOptionType::DOUBLE) {
	  if (check_option.value == value_pointer) {
	    HighsLogMessage(HighsMessageType::ERROR,
			    "checkOptions: Option %d (\"%s\") has the same value pointer as option %d (\"%s\")",
			    index, option.name.c_str(),
			    check_index, check_option.name.c_str());
	    error_found = true;
	  }
	}
      }
    } else if (type == HighsOptionType::STRING) {
      // Check string option
      OptionRecordString& option = ((OptionRecordString*)option_records[index])[0];
      // Check that there are no other options with the same value pointers
      std::string* value_pointer = option.value;
      for (int check_index = 0; check_index < num_options; check_index++) {
	if (check_index == index) continue;
	OptionRecordString& check_option = ((OptionRecordString*)option_records[check_index])[0];
	if (check_option.type == HighsOptionType::STRING) {
	  if (check_option.value == value_pointer) {
	    HighsLogMessage(HighsMessageType::ERROR,
			    "checkOptions: Option %d (\"%s\") has the same value pointer as option %d (\"%s\")",
			    index, option.name.c_str(),
			    check_index, check_option.name.c_str());
	    error_found = true;
	  }
	}
      }
    }
  }
  if (error_found) return OptionStatus::ILLEGAL_VALUE;
  HighsLogMessage(HighsMessageType::INFO, "checkOptions: Options are OK");
  return OptionStatus::OK;
}

OptionStatus checkOption(const OptionRecordInt& option) {
  if (option.lower_bound > option.upper_bound) {
    HighsLogMessage(HighsMessageType::ERROR, "checkOption: Option \"%s\" has inconsistent bounds [%d, %d]",
		    option.name.c_str(), option.lower_bound, option.upper_bound);
    return OptionStatus::ILLEGAL_VALUE;
  }
  if (option.default_value < option.lower_bound ||
      option.default_value > option.upper_bound) {
    HighsLogMessage(HighsMessageType::ERROR, "checkOption: Option \"%s\" has default value %d inconsistent with bounds [%d, %d]",
		    option.name.c_str(), option.default_value, option.lower_bound, option.upper_bound);
    return OptionStatus::ILLEGAL_VALUE;
  }
  int value = *option.value;
  if (value < option.lower_bound ||
      value > option.upper_bound) {
    HighsLogMessage(HighsMessageType::ERROR, "checkOption: Option \"%s\" has value %d inconsistent with bounds [%d, %d]",
		    option.name.c_str(), value, option.lower_bound, option.upper_bound);
    return OptionStatus::ILLEGAL_VALUE;
  }
  return OptionStatus::OK;
}

OptionStatus checkOption(const OptionRecordDouble& option) {
  if (option.lower_bound > option.upper_bound) {
    HighsLogMessage(HighsMessageType::ERROR, "checkOption: Option \"%s\" has inconsistent bounds [%g, %g]",
		    option.name.c_str(), option.lower_bound, option.upper_bound);
    return OptionStatus::ILLEGAL_VALUE;
  }
  if (option.default_value < option.lower_bound ||
      option.default_value > option.upper_bound) {
    HighsLogMessage(HighsMessageType::ERROR, "checkOption: Option \"%s\" has default value %g inconsistent with bounds [%g, %g]",
		    option.name.c_str(), option.default_value, option.lower_bound, option.upper_bound);
    return OptionStatus::ILLEGAL_VALUE;
  }
  double value = *option.value;
  if (value < option.lower_bound ||
      value > option.upper_bound) {
    HighsLogMessage(HighsMessageType::ERROR, "checkOption: Option \"%s\" has value %g inconsistent with bounds [%g, %g]",
		    option.name.c_str(), value, option.lower_bound, option.upper_bound);
    return OptionStatus::ILLEGAL_VALUE;
  }
  return OptionStatus::OK;
}

OptionStatus setOptionValue(const std::string& name, std::vector<OptionRecord*>& option_records, const bool value) {
  int index;
  //  printf("setOptionValue: \"%s\" with bool %d\n", name.c_str(), value);
  OptionStatus status = getOptionIndex(name, option_records, index);
  if (status != OptionStatus::OK) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::BOOL) {
    HighsLogMessage(HighsMessageType::ERROR, "setOptionValue: Option \"%s\" cannot be assigned a bool", name.c_str());
    return OptionStatus::ILLEGAL_VALUE;
  }
  return setOptionValue(((OptionRecordBool*)option_records[index])[0], value);
}

OptionStatus setOptionValue(const std::string& name, std::vector<OptionRecord*>& option_records, const int value) {
  int index;
  //  printf("setOptionValue: \"%s\" with int %d\n", name.c_str(), value);
  OptionStatus status = getOptionIndex(name, option_records, index);
  if (status != OptionStatus::OK) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::INT) {
    HighsLogMessage(HighsMessageType::ERROR, "setOptionValue: Option \"%s\" cannot be assigned an int", name.c_str());
    return OptionStatus::ILLEGAL_VALUE;
  }
  return setOptionValue(((OptionRecordInt*)option_records[index])[0], value);
}

// OptionStatus setFindFeasibilityStrategyValue(HighsOptions& options,
//                                              const std::string& value) {
//   if (value == "approx_component")
//     options.feasibility_strategy = FeasibilityStrategy::kApproxComponentWise;
//   else if (value == "approx_exact")
//     options.feasibility_strategy = FeasibilityStrategy::kApproxExact;
//   else if (value == "approx_breakpoints")
//     options.feasibility_strategy = FeasibilityStrategy::kApproxComponentWiseBreakpoints;
//   else if (value == "direct")
//     options.feasibility_strategy = FeasibilityStrategy::kDirectSolve;
//   else {
//     HighsLogMessage(HighsMessageType::ERROR,
//                     "feasibility component-wise value \"%s\" is not permitted: "
//                     "legal values are \"%s\" and \"%s\"\n",
//                     value.c_str(), "on", "off");
OptionStatus setOptionValue(const std::string& name, std::vector<OptionRecord*>& option_records, const double value) {
  int index;
  //  printf("setOptionValue: \"%s\" with double %g\n", name.c_str(), value);
  OptionStatus status = getOptionIndex(name, option_records, index);
  if (status != OptionStatus::OK) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::DOUBLE) {
    HighsLogMessage(HighsMessageType::ERROR, "setOptionValue: Option \"%s\" cannot be assigned a double", name.c_str());
    return OptionStatus::ILLEGAL_VALUE;
  }
  return setOptionValue(((OptionRecordDouble*)option_records[index])[0], value);
}

OptionStatus setOptionValue(const std::string& name, std::vector<OptionRecord*>& option_records, const std::string value) {
  int index;
  //  printf("setOptionValue: \"%s\" with value string %s\n", name.c_str(), value.c_str());
  OptionStatus status = getOptionIndex(name, option_records, index);
  if (status != OptionStatus::OK) return status;
  HighsOptionType type = option_records[index]->type;
  if (type == HighsOptionType::BOOL) {
    bool bool_value;
    bool return_status = boolFromString(value, bool_value);
    //    printf("boolFromString for \"%s\" returns %d from \"%s\" with status %d\n", name.c_str(), bool_value, value.c_str(), return_status);
    if (!return_status) {
      HighsLogMessage(HighsMessageType::ERROR, "setOptionValue: Value \"%s\" cannot be interpreted as a bool", value.c_str());
      return OptionStatus::ILLEGAL_VALUE;
    }
    return setOptionValue(((OptionRecordBool*)option_records[index])[0], bool_value);
  } else if (type == HighsOptionType::INT) {
    int value_int = atoi(value.c_str());
    double value_double = atof(value.c_str());
    double value_int_double = value_int;
    if (value_double != value_int_double) {
      HighsLogMessage(HighsMessageType::ERROR,
		      "setOptionValue: Value = \"%s\" converts via atoi as %d so is %g as double, but as %g via atof",
		      value.c_str(), value_int, value_int_double, value_double);
      return OptionStatus::ILLEGAL_VALUE;
    }
    return setOptionValue(((OptionRecordInt*)option_records[index])[0], value_int);
  } else if (type == HighsOptionType::DOUBLE) {
    int value_int = atoi(value.c_str());
    double value_double = atof(value.c_str());
    double value_int_double = value_int;
    if (value_double == value_int_double) {
      HighsLogMessage(HighsMessageType::INFO,
		      "setOptionValue: Value = \"%s\" converts via atoi as %d so is %g as double, and %g via atof\n",
		      value.c_str(), value_int, value_int_double, value_double);
    }
    return setOptionValue(((OptionRecordDouble*)option_records[index])[0], atof(value.c_str()));
  } else {
    return setOptionValue(((OptionRecordString*)option_records[index])[0], value);
  }
}

OptionStatus setOptionValue(const std::string& name, std::vector<OptionRecord*>& option_records, const char* value) {
  // Handles values passed as explicit values in quotes 
  std::string value_as_string(value);
  return setOptionValue(name, option_records, value_as_string);
}

OptionStatus setOptionValue(OptionRecordBool& option, const bool value) {
  option.assignvalue(value);
  return OptionStatus::OK;
}

OptionStatus setOptionValue(OptionRecordInt& option, const int value) {
  if (value < option.lower_bound) {
    HighsLogMessage(HighsMessageType::ERROR, "setOptionValue: Trying to set option \"%s\" to value %d below lower bound of %d",
	   option.name.c_str(), value, option.lower_bound);
    return OptionStatus::ILLEGAL_VALUE;
  } else if (value > option.upper_bound) {
    HighsLogMessage(HighsMessageType::ERROR, "setOptionValue: Trying to set option \"%s\" to value %d above upper bound of %d",
	   option.name.c_str(), value, option.upper_bound);
    return OptionStatus::ILLEGAL_VALUE;
  }
  option.assignvalue(value);
  return OptionStatus::OK;
}

OptionStatus setOptionValue(OptionRecordDouble& option, const double value) {
  if (value < option.lower_bound) {
    HighsLogMessage(HighsMessageType::ERROR, "setOptionValue: Trying to set option \"%s\" to value %g below lower bound of %g",
	   option.name.c_str(), value, option.lower_bound);
    return OptionStatus::ILLEGAL_VALUE;
  } else if (value > option.upper_bound) {
    HighsLogMessage(HighsMessageType::ERROR, "setOptionValue: Trying to set option \"%s\" to value %g above upper bound of %g",
	   option.name.c_str(), value, option.upper_bound);
    return OptionStatus::ILLEGAL_VALUE;
  }
  option.assignvalue(value);
  return OptionStatus::OK;
}

OptionStatus setOptionValue(OptionRecordString& option, const std::string value) {
  // Setting a string option: check that value is OK
  if (option.name == presolve_string) {
    if (!commandLineOffChooseOnOk(value)) return OptionStatus::ILLEGAL_VALUE;
  } else if (option.name == solver_string) {
    if (!commandLineSolverOk(value)) return OptionStatus::ILLEGAL_VALUE;
  } else if (option.name == parallel_string) {
    if (!commandLineOffChooseOnOk(value)) return OptionStatus::ILLEGAL_VALUE;
  }
  option.assignvalue(value);
  return OptionStatus::OK;
}

OptionStatus getOptionValue(const std::string& name, const std::vector<OptionRecord*>& option_records, bool& value) {
  int index;
  OptionStatus status = getOptionIndex(name, option_records, index);
  if (status != OptionStatus::OK) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::BOOL) {
    HighsLogMessage(HighsMessageType::ERROR, "getOptionValue: Option \"%s\" requires value of type %s, not bool", name.c_str(), optionEntryType2string(type).c_str());
    return OptionStatus::ILLEGAL_VALUE;
  }
  OptionRecordBool option = ((OptionRecordBool*)option_records[index])[0];
  value = *option.value;
  return OptionStatus::OK;
}

OptionStatus getOptionValue(const std::string& name, const std::vector<OptionRecord*>& option_records, int& value) {
  int index;
  OptionStatus status = getOptionIndex(name, option_records, index);
  if (status != OptionStatus::OK) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::INT) {
    HighsLogMessage(HighsMessageType::ERROR, "getOptionValue: Option \"%s\" requires value of type %s, not int", name.c_str(), optionEntryType2string(type).c_str());
    return OptionStatus::ILLEGAL_VALUE;
  }
  OptionRecordInt option = ((OptionRecordInt*)option_records[index])[0];
  value = *option.value;
  return OptionStatus::OK;
}

OptionStatus getOptionValue(const std::string& name, const std::vector<OptionRecord*>& option_records, double& value) {
  int index;
  OptionStatus status = getOptionIndex(name, option_records, index);
  if (status != OptionStatus::OK) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::DOUBLE) {
    HighsLogMessage(HighsMessageType::ERROR, "getOptionValue: Option \"%s\" requires value of type %s, not double", name.c_str(), optionEntryType2string(type).c_str());
    return OptionStatus::ILLEGAL_VALUE;
  }
  OptionRecordDouble option = ((OptionRecordDouble*)option_records[index])[0];
  value = *option.value;
  return OptionStatus::OK;
}

OptionStatus getOptionValue(const std::string& name, const std::vector<OptionRecord*>& option_records, std::string& value) {
  int index;
  OptionStatus status = getOptionIndex(name, option_records, index);
  if (status != OptionStatus::OK) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::STRING) {
    HighsLogMessage(HighsMessageType::ERROR, "getOptionValue: Option \"%s\" requires value of type %s, not string", name.c_str(), optionEntryType2string(type).c_str());
    return OptionStatus::ILLEGAL_VALUE;
  }
  OptionRecordString option = ((OptionRecordString*)option_records[index])[0];
  value = *option.value;
  return OptionStatus::OK;
}

HighsStatus reportOptionsToFile(const std::string filename, const std::vector<OptionRecord*>& option_records) {
  FILE* file = fopen(filename.c_str(), "w");
  if (file == 0) {
    HighsLogMessage(HighsMessageType::ERROR, "reportOptionsToFile: cannot open file");
    return HighsStatus::Error;
  }
  reportOptions(file, option_records, true);
  return HighsStatus::OK;
}

void reportOptions(FILE* file, const std::vector<OptionRecord*>& option_records, const bool force_report) {
  int num_options = option_records.size();
  for (int index = 0; index < num_options; index++) {
    HighsOptionType type = option_records[index]->type;
    //    fprintf(file, "\n# Option %1d\n", index);
    if (type == HighsOptionType::BOOL) {
      reportOption(file, ((OptionRecordBool*)option_records[index])[0], force_report);
    } else if (type == HighsOptionType::INT) {
      reportOption(file, ((OptionRecordInt*)option_records[index])[0], force_report);
    } else if (type == HighsOptionType::DOUBLE) {
      reportOption(file, ((OptionRecordDouble*)option_records[index])[0], force_report);
    } else {
      reportOption(file, ((OptionRecordString*)option_records[index])[0], force_report);
    } 
  }
}

void reportOption(FILE* file, const OptionRecordBool& option, const bool force_report) {
  if (force_report || option.default_value != *option.value) {
    fprintf(file, "\n# %s\n", option.description.c_str());
    fprintf(file, "# [type: bool, advanced: %s, range: {false, true}, default: %s]\n",
	   bool2string(option.advanced),
	   bool2string(option.default_value));
    fprintf(file, "%s = %s\n", option.name.c_str(), bool2string(*option.value));
  }
}

void reportOption(FILE* file, const OptionRecordInt& option, const bool force_report) {
  if (force_report || option.default_value != *option.value) {
    fprintf(file, "\n# %s\n", option.description.c_str());
    fprintf(file, "# [type: int, advanced: %s, range: {%d, %d}, default: %d]\n",
	   bool2string(option.advanced),
	   option.lower_bound,
	   option.upper_bound,
	   option.default_value);
    fprintf(file, "%s = %d\n", option.name.c_str(), *option.value);
  }
}

void reportOption(FILE* file, const OptionRecordDouble& option, const bool force_report) {
  if (force_report || option.default_value != *option.value) {
    fprintf(file, "\n# %s\n", option.description.c_str());
    fprintf(file, "# [type: double, advanced: %s, range: [%g, %g], default: %g]\n",
	   bool2string(option.advanced),
	   option.lower_bound,
	   option.upper_bound,
	   option.default_value);
    fprintf(file, "%s = %g\n", option.name.c_str(), *option.value);
  }
}

void reportOption(FILE* file, const OptionRecordString& option, const bool force_report) {
  // Don't report for the options file if writing to an options file
  if (
      //file != stdout &&
      option.name == options_file_string) return;
  if (force_report || option.default_value != *option.value) {
    fprintf(file, "\n# %s\n", option.description.c_str());
    fprintf(file, "# [type: string, advanced: %s, default: \"%s\"]\n",
	   bool2string(option.advanced),
	   option.default_value.c_str());
    fprintf(file, "%s = %s\n", option.name.c_str(), (*option.value).c_str());
  }
}

OptionStatus checkOptionsValue(HighsOptions& options) {
  return OptionStatus::OK;
}

// Set values of options so that HiGHS runs as Hsol
void setHsolOptions(HighsOptions& options) {
  // Set command line options to their hsol values
  options.presolve = OPTION_OFF;
  options.solver = SOLVER_OPTION_SIMPLEX;
  options.parallel = OPTION_OFF;
  options.time_limit = HIGHS_CONST_INF;
  
  options.simplex_iteration_limit = HIGHS_CONST_I_INF;
  options.mps_parser_type_free = false;
  options.keep_n_rows = KEEP_N_ROWS_KEEP_ROWS;
  options.infinite_cost = HIGHS_CONST_INF;
  options.infinite_bound = HIGHS_CONST_INF;
  options.small_matrix_value = 0;
  options.large_matrix_value = HIGHS_CONST_INF;
  options.allowed_simplex_scale_factor = HIGHS_CONST_I_INF;
  options.primal_feasibility_tolerance = 1e-7;
  options.dual_feasibility_tolerance = 1e-7;
  options.dual_objective_value_upper_bound = HIGHS_CONST_INF;
  options.simplex_strategy = SIMPLEX_STRATEGY_DUAL_PLAIN;
  options.simplex_dualise_strategy = OPTION_OFF;
  options.simplex_permute_strategy = OPTION_OFF;
  options.simplex_scale_strategy = SIMPLEX_SCALE_STRATEGY_HSOL;
  options.simplex_crash_strategy = SIMPLEX_CRASH_STRATEGY_OFF;
  options.simplex_dual_edge_weight_strategy = SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE;
  options.simplex_primal_edge_weight_strategy = SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY_DANTZIG;
  options.simplex_price_strategy = SIMPLEX_PRICE_STRATEGY_ROW;
  options.less_infeasible_DSE_check = false;
}

OptionStatus setMessageLevelValue(HighsOptions& options, const int& value) {
  HighsSetMessagelevel(value);
  options.message_level = value;
  return OptionStatus::OK;
}
