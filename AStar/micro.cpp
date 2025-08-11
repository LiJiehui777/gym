// This file is used for training, do not submit.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <bitset>
#include <iostream>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

static constexpr short moves[5] = {3, 1, -5, 0, 2};
static constexpr short dirs[5][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {0, 0}};

class PQ : public std::priority_queue<
               std::tuple<short, short, short>,
               std::vector<std::tuple<short, short, short>>,
               std::greater<std::tuple<short, short, short>>> {
 public:
  void clear() { this->c.clear(); }
};

class Player {
 public:
  Player() : pos_r(0), pos_c(0), initialized(false) {}

  static Player& getInstance() {
    static Player instance;
    return instance;
  }

  std::vector<int> MoveDecision(const std::vector<std::vector<int>>& grid,
                                int gold, int gold_2) {
    std::vector<std::vector<int>>& changableGrid =
        const_cast<std::vector<std::vector<int>>&>(grid);

    if (!initialized) {
      init(grid);
      initialized = true;
    }
    initPosition(grid);
    this->gold_ = gold;
    short r = pos_r, c = pos_c;

    _MoveDecision(changableGrid, gold, gold_2);

    std::vector<int> ans{directions[0], directions[1], directions[2]};
    return ans;
  }

 private:
  void _MoveDecision(std::vector<std::vector<int>>& grid, int gold,
                     int gold_2) {
    directionsCnt = 0;
    while (directionsCnt < 3) {
      short target = chooseCoin(grid);
      if (target == -1) {
        break;
      }
      aStar(grid, target);
    }
    while (directionsCnt < 3) {
      directions[directionsCnt++] = 4;
    }
  }

  inline void aStar(std::vector<std::vector<int>>& grid, short target)
      __attribute__((always_inline)) {
    short startHashVal = (pos_r << 4) + pos_r + pos_c;
    pq.clear();
    pq.emplace(0, startHashVal, 0);

    std::memset(dis, 0x7f, sizeof(dis));
    dis[startHashVal] = 0;
    visited.reset();
    visited.set(startHashVal);

    while (!pq.empty()) {
      auto node = pq.top();
      pq.pop();
      short x = std::get<1>(node);
      short cost = std::get<2>(node);

      if (cost > dis[x]) {
        continue;
      } else if (x == target) {
        break;
      }

      short r = x / 17;
      short c = x - (r << 4) - r;
      visited.set(x);

      addNode(pq, r, c + 1, cost, target, x, grid);
      addNode(pq, r, c - 1, cost, target, x, grid);
      addNode(pq, r + 1, c, cost, target, x, grid);
      addNode(pq, r - 1, c, cost, target, x, grid);
    }

    paths.clear();
    paths.emplace_back(target);
    while (target != startHashVal) {
      target = parents[target];
      paths.emplace_back(target);
    }

    for (int i = paths.size() - 2; i >= 0 && directionsCnt < 3; --i) {
      short next_r = paths[i] / 17;
      short next_c = paths[i] - (next_r << 4) - next_r;
      directions[directionsCnt++] =
          moves[pos_r - next_r + (pos_c - next_c) * 2 + 2];
      pos_r = next_r;
      pos_c = next_c;
      grid[pos_r][pos_c] = 0;
    }
  }

  inline void addNode(PQ& pd, short new_r, short new_c, short cost,
                      short target, short parentNode,
                      const std::vector<std::vector<int>>& grid)
      __attribute__((always_inline)) {
    short new_pos = (new_r << 4) + new_r + new_c;
    if (new_r < 0 || new_r >= 17 || new_c < 0 || new_c >= 17 ||
        grid[new_r][new_c] == -1 || visited.test(new_pos)) {
      return;
    }
    short new_d = cost + 1;
    if (grid[new_r][new_c] > 0) {
      new_d -= std::min(new_d, static_cast<short>(grid[new_r][new_c]));
    } else if (grid[new_r][new_c] == -3) {
      new_d += gold_ * 0.3;  // bomb
    }
    if (new_d < dis[new_pos]) {
      parents[new_pos] = parentNode;
      dis[new_pos] = new_d;
      pq.emplace(new_d + minDistances[target][new_pos], new_pos, new_d);
    }
  }

  inline void calcMinDistances(const std::vector<std::vector<int>>& grid, int r,
                               int c) __attribute__((always_inline)) {
    // TODO: maybe this function (BFS) can be speed up.
    short startHashVal = (r << 4) + r + c;
    visited.reset();
    visited.set(startHashVal);
    auto& depth = minDistances[startHashVal];
    depth[startHashVal] = 0;
    // Use variable `parents` as a queue.
    parents[0] = startHashVal;
    short cnt = 1;

    for (int idx = 0; idx < cnt; ++idx) {
      short hashVal = parents[idx];
      short r = hashVal / 17, c = hashVal - (r << 4) - r;

      addBFSNode(r, c + 1, grid, cnt, startHashVal, hashVal);
      addBFSNode(r, c - 1, grid, cnt, startHashVal, hashVal);
      addBFSNode(r + 1, c, grid, cnt, startHashVal, hashVal);
      addBFSNode(r - 1, c, grid, cnt, startHashVal, hashVal);
    }
  }

  inline void addBFSNode(short new_r, short new_c,
                         const std::vector<std::vector<int>>& grid, short& cnt,
                         short startHashVal, short hashVal)
      __attribute__((always_inline)) {
    if (new_r < 0 || new_r >= 17 || new_c < 0 || new_c >= 17 ||
        grid[new_r][new_c] == -1) {
      return;
    }
    short newHashVal = (new_r << 4) + new_r + new_c;
    if (visited.test(newHashVal)) {
      return;
    }
    visited.set(newHashVal);
    // Use variable `parents` as a queue.
    parents[cnt++] = newHashVal;
    minDistances[startHashVal][newHashVal] =
        minDistances[startHashVal][hashVal] + 1;
  }

  inline void init(const std::vector<std::vector<int>>& grid)
      __attribute__((always_inline)) {
    // All the initializations are performed here.
    for (int r = 0; r < 17; ++r) {
      for (int c = 0; c < 17; ++c) {
        if (grid[r][c] != -1) {
          calcMinDistances(grid, r, c);
        }
      }
    }
  }

  inline short chooseCoin(std::vector<std::vector<int>>& grid)
      __attribute__((always_inline)) {
    float maxScore = -1e10;
    short target = -1;
    short posHash = (pos_r << 4) + pos_r + pos_c;

    for (short pos = 0; pos < 289; ++pos) {
      int r = pos / 17, c = pos - (r << 4) - r;
      if (grid[r][c] <= 0) {
        continue;
      }
      float score = grid[r][c] / (1.0 + 2 * minDistances[posHash][pos]);
      if (score > maxScore) {
        maxScore = score;
        target = pos;
      }
    }

    return target;
  }

  inline void initPosition(const std::vector<std::vector<int>>& grid)
      __attribute__((always_inline)) {
    if (grid[pos_r][pos_c] == -9) {
      return;
    }
    for (short pos = 0; pos < 289; ++pos) {
      short r = pos / 17, c = pos - (r << 4) - r;
      if (grid[r][c] == -9) {
        pos_r = r;
        pos_c = c;
        break;
      }
    }
  }

 private:
  short pos_r, pos_c;
  short parents[289];
  short dis[289];
  std::bitset<289> visited;
  int gold_;
  char minDistances[289][289];
  bool initialized;
  PQ pq;
  std::vector<short> paths;

  int directions[3] = {4, 4, 4};
  int directionsCnt;
};

PYBIND11_MODULE(micro, m) {
  py::class_<Player>(m, "Player", py::module_local())
      .def(py::init<>())
      .def("MoveDecision", &Player::MoveDecision);
}
