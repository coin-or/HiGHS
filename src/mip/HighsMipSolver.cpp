/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2021 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsMipSolver.h"

#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsModelUtils.h"
#include "mip/HighsCliqueTable.h"
#include "mip/HighsCutPool.h"
#include "mip/HighsDomain.h"
#include "mip/HighsImplications.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsPseudocost.h"
#include "mip/HighsSearch.h"
#include "mip/HighsSeparation.h"
#include "presolve/HPresolve.h"
#include "presolve/HighsPostsolveStack.h"
#include "presolve/PresolveComponent.h"
#include "util/HighsCDouble.h"

HighsMipSolver::HighsMipSolver(const HighsOptions& options, const HighsLp& lp,
                               bool submip)
    : options_mip_(&options),
      model_(&lp),
      solution_objective_(HIGHS_CONST_INF),
      submip(submip),
      rootbasis(nullptr) {}

HighsMipSolver::~HighsMipSolver() = default;

HighsPresolveStatus HighsMipSolver::runPresolve() {
  // todo: commented out parts or change Highs::runPresolve to operate on a
  // parameter LP rather than Highs::lp_. Not sure which approach is preferable.
  highsLogDev(options_mip_->log_options, HighsLogType::INFO,
              "\nrunning MIP presolve\n");
  const HighsLp& lp_ = *(model_);

  // Exit if the problem is empty or if presolve is set to off.
  if (options_mip_->presolve == off_string)
    return HighsPresolveStatus::NotPresolved;
  if (lp_.numCol_ == 0 && lp_.numRow_ == 0)
    return HighsPresolveStatus::NullError;

  // Clear info from previous runs if lp_ has been modified.
  if (presolve_.has_run_) presolve_.clear();
  double start_presolve = timer_.readRunHighsClock();

  // Set time limit.
  if (options_mip_->time_limit > 0 &&
      options_mip_->time_limit < HIGHS_CONST_INF) {
    double left = options_mip_->time_limit - start_presolve;
    if (left <= 0) {
      highsLogDev(options_mip_->log_options, HighsLogType::VERBOSE,
                  "Time limit reached while reading in matrix\n");
      return HighsPresolveStatus::Timeout;
    }

    highsLogDev(options_mip_->log_options, HighsLogType::VERBOSE,
                "Time limit set: reading matrix took %.2g, presolve "
                "time left: %.2g\n",
                start_presolve, left);
    presolve_.options_.time_limit = left;
  }

  // Presolve.
  presolve_.init(lp_, timer_, true);

  if (options_mip_->time_limit > 0 &&
      options_mip_->time_limit < HIGHS_CONST_INF) {
    double current = timer_.readRunHighsClock();
    double time_init = current - start_presolve;
    double left = presolve_.options_.time_limit - time_init;
    if (left <= 0) {
      highsLogDev(options_mip_->log_options, HighsLogType::VERBOSE,
                  "Time limit reached while copying matrix into presolve.\n");
      return HighsPresolveStatus::Timeout;
    }

    highsLogDev(options_mip_->log_options, HighsLogType::VERBOSE,
                "Time limit set: copying matrix took %.2g, presolve "
                "time left: %.2g\n",
                time_init, left);
    presolve_.options_.time_limit = options_mip_->time_limit;
  }

  presolve_.data_.presolve_[0].log_options = options_mip_->log_options;

  HighsPresolveStatus presolve_return_status = presolve_.run();

  // Handle max case.
  if (presolve_return_status == HighsPresolveStatus::Reduced &&
      lp_.sense_ == ObjSense::MAXIMIZE) {
    presolve_.negateReducedLpCost();
    presolve_.data_.reduced_lp_.sense_ = ObjSense::MAXIMIZE;
  }

  // Update reduction counts.
  switch (presolve_.presolve_status_) {
    case HighsPresolveStatus::Reduced: {
      HighsLp& reduced_lp = presolve_.getReducedProblem();
      presolve_.info_.n_cols_removed = lp_.numCol_ - reduced_lp.numCol_;
      presolve_.info_.n_rows_removed = lp_.numRow_ - reduced_lp.numRow_;
      presolve_.info_.n_nnz_removed =
          (int)lp_.Avalue_.size() - (int)reduced_lp.Avalue_.size();
      break;
    }
    case HighsPresolveStatus::ReducedToEmpty: {
      presolve_.info_.n_cols_removed = lp_.numCol_;
      presolve_.info_.n_rows_removed = lp_.numRow_;
      presolve_.info_.n_nnz_removed = (int)lp_.Avalue_.size();
      break;
    }
    default:
      break;
  }
  return presolve_return_status;
}

void HighsMipSolver::run() {
  modelstatus_ = HighsModelStatus::NOTSET;
  // std::cout << options_mip_->presolve << std::endl;
  timer_.start(timer_.solve_clock);
  timer_.start(timer_.presolve_clock);

  mipdata_ = decltype(mipdata_)(new HighsMipSolverData(*this));
  mipdata_->init();
  mipdata_->runPresolve();
  if (modelstatus_ != HighsModelStatus::NOTSET) {
    highsLogUser(options_mip_->log_options, HighsLogType::INFO,
                 "Presolve: %s\n",
                 utilHighsModelStatusToString(modelstatus_).c_str());
    if (modelstatus_ == HighsModelStatus::OPTIMAL) {
      mipdata_->lower_bound = 0;
      mipdata_->upper_bound = 0;
      mipdata_->transformNewIncumbent(std::vector<double>());
    }
    timer_.stop(timer_.presolve_clock);
    cleanupSolve();
    return;
  }
#if 0
  if (options_mip_->presolve != "off") {
    HighsPresolveStatus presolve_status = runPresolve();
    switch (presolve_status) {
      case HighsPresolveStatus::Reduced:
        reportPresolveReductions(options_mip_->log_options, *model_,
                                 presolve_.getReducedProblem());
        model_ = &presolve_.getReducedProblem();
        break;
      case HighsPresolveStatus::Unbounded:
        modelstatus_ = HighsModelStatus::PRIMAL_UNBOUNDED;
        highsLogUser(options_mip_->log_options, HighsLogType::INFO,
                     "Presolve: Model detected to be unbounded\n");
        timer_.stop(timer_.presolve_clock);
        timer_.stop(timer_.solve_clock);
        return;
      case HighsPresolveStatus::Infeasible:
        modelstatus_ = HighsModelStatus::PRIMAL_INFEASIBLE;
        highsLogUser(options_mip_->log_options, HighsLogType::INFO,
                     "Presolve: Model detected to be infeasible\n");
        timer_.stop(timer_.presolve_clock);
        timer_.stop(timer_.solve_clock);
        return;
      case HighsPresolveStatus::Timeout:
        modelstatus_ = HighsModelStatus::REACHED_TIME_LIMIT;
        highsLogUser(options_mip_->log_options, HighsLogType::INFO,
                     "Time limit reached during presolve\n");
        timer_.stop(timer_.presolve_clock);
        timer_.stop(timer_.solve_clock);
        return;
      case HighsPresolveStatus::ReducedToEmpty:
        modelstatus_ = HighsModelStatus::OPTIMAL;
        reportPresolveReductions(options_mip_->log_options, *model_, true);
        mipdata_ = decltype(mipdata_)(new HighsMipSolverData(*this));
        mipdata_->init();
        mipdata_->upper_bound = presolve_.data_.presolve_[0].objShift;
        mipdata_->lower_bound = presolve_.data_.presolve_[0].objShift;
        timer_.stop(timer_.presolve_clock);
        cleanupSolve();
        return;
      case HighsPresolveStatus::NotReduced:
        reportPresolveReductions(options_mip_->log_options, *model_, false);
        break;
      default:
        assert(false);
    }
  }
#endif

  mipdata_->runSetup();
  timer_.stop(timer_.presolve_clock);
  if (modelstatus_ == HighsModelStatus::NOTSET) {
    mipdata_->evaluateRootNode();
  }
  if (mipdata_->nodequeue.empty()) {
    cleanupSolve();
    return;
  }

  std::shared_ptr<const HighsBasis> basis;
  HighsSearch search{*this, mipdata_->pseudocost};
  mipdata_->debugSolution.registerDomain(search.getLocalDomain());
  HighsSeparation sepa(*this);

  search.setLpRelaxation(&mipdata_->lp);
  sepa.setLpRelaxation(&mipdata_->lp);

  mipdata_->lower_bound = mipdata_->nodequeue.getBestLowerBound();

  highsLogUser(options_mip_->log_options, HighsLogType::INFO,
               "\nstarting tree search\n");
  mipdata_->printDisplayLine();
  search.installNode(mipdata_->nodequeue.popBestBoundNode());

  while (search.hasNode()) {
    // set iteration limit for each lp solve during the dive to 10 times the
    // average nodes

    int iterlimit =
        10 *
        int(mipdata_->total_lp_iterations - mipdata_->sb_lp_iterations -
            mipdata_->sepa_lp_iterations) /
        (double)std::max(size_t{1}, mipdata_->num_nodes);
    iterlimit = std::max(10000, iterlimit);

    mipdata_->lp.setIterationLimit(iterlimit);

    // perform the dive and put the open nodes to the queue
    size_t plungestart = mipdata_->num_nodes;
    size_t lastLbLeave = mipdata_->num_leaves;
    bool limit_reached = false;
    while (true) {
      if (mipdata_->heuristic_lp_iterations <
          mipdata_->total_lp_iterations * mipdata_->heuristic_effort) {
        search.evaluateNode();
        if (search.currentNodePruned()) {
          ++mipdata_->num_leaves;
          search.flushStatistics();
          break;
        }

        if (mipdata_->incumbent.empty())
          mipdata_->heuristics.randomizedRounding(
              mipdata_->lp.getLpSolver().getSolution().col_value);

        if (mipdata_->incumbent.empty())
          mipdata_->heuristics.RENS(
              mipdata_->lp.getLpSolver().getSolution().col_value);
        else
          mipdata_->heuristics.RINS(
              mipdata_->lp.getLpSolver().getSolution().col_value);

        mipdata_->heuristics.flushStatistics();
      }

      if (mipdata_->domain.infeasible()) break;

      search.dive();
      ++mipdata_->num_leaves;

      search.flushStatistics();
      if (mipdata_->checkLimits()) {
        limit_reached = true;
        break;
      }

      if (!search.backtrack()) break;

      if (mipdata_->upper_limit == HIGHS_CONST_INF ||
          search.getCurrentEstimate() >= mipdata_->upper_limit)
        break;

      if (mipdata_->num_nodes - plungestart >=
          std::min(size_t{100}, mipdata_->num_nodes / 10))
        break;

      if (mipdata_->dispfreq != 0) {
        if (mipdata_->num_leaves - mipdata_->last_displeave >=
            std::min(size_t{mipdata_->dispfreq},
                     1 + size_t(0.01 * mipdata_->num_leaves)))
          mipdata_->printDisplayLine();
      }

      // printf("continue plunging due to good esitmate\n");
    }
    search.openNodesToQueue(mipdata_->nodequeue);
    mipdata_->lower_bound = std::min(mipdata_->upper_bound,
                                     mipdata_->nodequeue.getBestLowerBound());

    if (limit_reached) break;

    if (mipdata_->dispfreq != 0) {
      if (mipdata_->num_leaves - mipdata_->last_displeave >=
          std::min(size_t{mipdata_->dispfreq},
                   1 + size_t(0.01 * mipdata_->num_leaves)))
        mipdata_->printDisplayLine();
    }

    // the search datastructure should have no installed node now
    assert(!search.hasNode());

    // propagate the global domain
    mipdata_->domain.propagate();
    mipdata_->pruned_treeweight += mipdata_->nodequeue.pruneInfeasibleNodes(
        mipdata_->domain, mipdata_->feastol);

    // if global propagation detected infeasibility, stop here
    if (mipdata_->domain.infeasible()) {
      mipdata_->nodequeue.clear();
      mipdata_->pruned_treeweight = 1.0;
      mipdata_->lower_bound = std::min(HIGHS_CONST_INF, mipdata_->upper_bound);
      break;
    }

    // if global propagation found bound changes, we update the local domain
    if (!mipdata_->domain.getChangedCols().empty()) {
      highsLogDev(options_mip_->log_options, HighsLogType::INFO,
                  "added %d global bound changes\n",
                  (int)mipdata_->domain.getChangedCols().size());
      mipdata_->cliquetable.cleanupFixed(mipdata_->domain);
      for (int col : mipdata_->domain.getChangedCols())
        mipdata_->implications.cleanupVarbounds(col);

      mipdata_->domain.setDomainChangeStack(std::vector<HighsDomainChange>());
      search.resetLocalDomain();

      mipdata_->domain.clearChangedCols();
      mipdata_->removeFixedIndices();
    }

    // remove the iteration limit when installing a new node
    // mipdata_->lp.setIterationLimit();

    // loop to install the next node for the search
    while (!mipdata_->nodequeue.empty()) {
      // printf("popping node from nodequeue (length = %d)\n",
      // (int)nodequeue.size());
      assert(!search.hasNode());

      if (mipdata_->num_leaves - lastLbLeave >= 10) {
        search.installNode(mipdata_->nodequeue.popBestBoundNode());
        lastLbLeave = mipdata_->num_leaves;
      } else {
        search.installNode(mipdata_->nodequeue.popBestNode());
        if (search.getCurrentLowerBound() == mipdata_->lower_bound)
          lastLbLeave = mipdata_->num_leaves;
      }

      assert(search.hasNode());

      // set the current basis if available
      if (basis) {
        mipdata_->lp.setStoredBasis(basis);
        mipdata_->lp.recoverBasis();
      }

      // we evaluate the node directly here instead of performing a dive
      // because we first want to check if the node is not fathomed due to
      // new global information before we perform separation rounds for the node
      search.evaluateNode();

      // if the node was pruned we remove it from the search and install the
      // next node from the queue
      if (search.currentNodePruned()) {
        search.backtrack();
        ++mipdata_->num_leaves;
        ++mipdata_->num_nodes;
        search.flushStatistics();

        if (mipdata_->domain.infeasible()) {
          mipdata_->nodequeue.clear();
          mipdata_->pruned_treeweight = 1.0;
          mipdata_->lower_bound =
              std::min(HIGHS_CONST_INF, mipdata_->upper_bound);
          break;
        }

        if (mipdata_->checkLimits()) {
          limit_reached = true;
          break;
        }

        mipdata_->lower_bound = std::min(
            mipdata_->upper_bound, mipdata_->nodequeue.getBestLowerBound());

        if (mipdata_->dispfreq != 0) {
          if (mipdata_->num_leaves - mipdata_->last_displeave >=
              std::min(size_t{mipdata_->dispfreq},
                       1 + size_t(0.01 * mipdata_->num_leaves)))
            mipdata_->printDisplayLine();
        }
        continue;
      }

      // the node is still not fathomed, so perform separation
      sepa.separate(search.getLocalDomain());

      // after separation we store the new basis and proceed with the outer loop
      // to perform a dive from this node
      if (mipdata_->lp.getStatus() != HighsLpRelaxation::Status::Error &&
          mipdata_->lp.getStatus() != HighsLpRelaxation::Status::NotSet)
        mipdata_->lp.storeBasis();

      basis = mipdata_->lp.getStoredBasis();

      break;
    }

    if (limit_reached) break;
  }

  cleanupSolve();
}

void HighsMipSolver::cleanupSolve() {
  timer_.start(timer_.postsolve_clock);
  bool havesolution = solution_objective_ != HIGHS_CONST_INF;
  dual_bound_ = mipdata_->lower_bound + model_->offset_;
  primal_bound_ = mipdata_->upper_bound + model_->offset_;
  node_count_ = mipdata_->num_nodes;

  if (modelstatus_ == HighsModelStatus::NOTSET) {
    if (havesolution)
      modelstatus_ = HighsModelStatus::OPTIMAL;
    else
      modelstatus_ = HighsModelStatus::PRIMAL_INFEASIBLE;
  }

  model_ = orig_model_;
  timer_.stop(timer_.postsolve_clock);
  timer_.stop(timer_.solve_clock);

  std::string solutionstatus = "-";

  if (havesolution) {
    bool feasible =
        bound_violation_ <= options_mip_->mip_feasibility_tolerance &&
        integrality_violation_ <= options_mip_->mip_feasibility_tolerance &&
        row_violation_ <= options_mip_->mip_feasibility_tolerance;
    solutionstatus = feasible ? "feasible" : "infeasible";
  }
  highsLogUser(options_mip_->log_options, HighsLogType::INFO,
               "\nSolving report\n"
               "  Status            %s\n"
               "  Primal bound      %.12g\n"
               "  Dual bound        %.12g\n"
               "  Solution status   %s\n",
               utilHighsModelStatusToString(modelstatus_).c_str(),
               primal_bound_, dual_bound_, solutionstatus.c_str());
  if (solutionstatus != "-")
    highsLogUser(options_mip_->log_options, HighsLogType::INFO,
                 "                    %.12g (objective)\n"
                 "                    %.12g (bound viol.)\n"
                 "                    %.12g (int. viol.)\n"
                 "                    %.12g (row viol.)\n",
                 solution_objective_, bound_violation_, integrality_violation_,
                 row_violation_);
  highsLogUser(options_mip_->log_options, HighsLogType::INFO,
               "  Timing            %.2f (total)\n"
               "                    %.2f (presolve)\n"
               "                    %.2f (postsolve)\n"
               "  Nodes             %llu\n"
               "  LP iterations     %llu (total)\n"
               "                    %llu (strong br.)\n"
               "                    %llu (separation)\n"
               "                    %llu (heuristics)\n",
               timer_.read(timer_.solve_clock),
               timer_.read(timer_.presolve_clock),
               timer_.read(timer_.postsolve_clock),
               (long long unsigned)mipdata_->num_nodes,
               (long long unsigned)mipdata_->total_lp_iterations,
               (long long unsigned)mipdata_->sb_lp_iterations,
               (long long unsigned)mipdata_->sepa_lp_iterations,
               (long long unsigned)mipdata_->heuristic_lp_iterations);

  assert(modelstatus_ != HighsModelStatus::NOTSET);
}
