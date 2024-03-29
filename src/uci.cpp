
#include <iostream>
#include <algorithm>
#include <cmath>
#include "uci.h"
#include "useful.h"
#include "move.h"
#include "perft.h"
#include "bench.h"

void UCI::initialize_uci() const {

    Position& position = mcts_engine->position;
    position.set_fen(START_FEN);
    mcts_engine->new_game();
}


void  UCI::time_handler(double self_time, double inc, double movetime, long movestogo) const {
    double rate = 20;
    double time_amt;

    Position& position = mcts_engine->position;

    if (position.is_attacked(position.get_king_pos(position.side), position.side)) rate -= 3;
    if (last_move.is_capture(position)) rate -= 1.5;

    if (movetime > 0) time_amt = movetime * 0.9;
    else if (inc > 0 && movestogo > 0) {
        time_amt = (self_time * 0.8 / static_cast<double>(movestogo)) * (20 / rate) + (inc / 2.0);
        if (time_amt > self_time * 0.8) time_amt = self_time * 0.85 + (inc / 2.0);
    }
    else if (inc > 0) {

        // we always want to have more time than our increment.
        // This ensures we use a lot of our remaining time, but
        // since our increment is larger, it doesn't matter.
        if (self_time < inc) time_amt = self_time / (rate / 10);
        else {
            // If our remaining time is less than the boundary, we should use less time than our increment
            // to get back above the boundary.
            double bound = inc / 2.5 * sqrt(60000.0 / inc);
            if (inc > bound / 2.5) bound = inc * sqrt(90000.0 / inc);
            if (inc > bound / 2.5) bound = 1.5 * inc * sqrt(200000.0 / inc);
            time_amt = std::max(inc * 0.975 + (self_time - bound) / (rate * 1.8), self_time / (rate * 10));
        }
    }
    else if (movestogo > 0) {
        time_amt = (self_time * 0.9 / static_cast<double>(movestogo)) * (20 / rate);
        if (time_amt > self_time * 0.9) time_amt = self_time * 0.95;
    }
    else if (self_time > 0) time_amt = self_time / (rate + 6);
    else time_amt = static_cast<double>(mcts_engine->max_time);

    mcts_engine->max_time = static_cast<uint64_t>(time_amt * 0.6);

    // std::cout << time_amt << " " << engine->hard_time_limit << " " << engine->soft_time_limit << std::endl;
}


void UCI::parse_position() {
    if (tokens.size() < 2) return;

    Position& position = mcts_engine->position;

    int next_idx;

    if (tokens[1] == "startpos") {
        mcts_engine->fifty_move = position.set_fen(START_FEN);
        next_idx = 2;
    }

    else if (tokens[1] == "fen") {
        std::string fen;
        for (int i = 2; i < 8; i++) {
            fen += tokens[i];
            fen += " ";
        }

        mcts_engine->fifty_move = position.set_fen(fen);
        next_idx = 8;
    }

    else return;

    mcts_engine->main_game_hashes.clear();

    if (static_cast<int>(tokens.size()) <= next_idx || tokens[next_idx] != "moves") return;

    for (int i = next_idx + 1; i < static_cast<int>(tokens.size()); i++) {
        mcts_engine->main_game_hashes.insert(position.hash_key);

        Move move = Move(position, tokens[i]);

        position.make_move(move, position.state_stack[0], mcts_engine->fifty_move);
        last_move = move;

        if (i - (next_idx + 1) >= current_move_idx) {
            std::cout << tokens[i] << std::endl;
            mcts_engine->update_tree(move);
            current_move_idx++;
        }
    }
}


void UCI::parse_go() {

    Position& position = mcts_engine->position;

    PLY_TYPE d = 0, perft_depth = 0;
    double wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0;
    long movestogo = 0;
    bool infinite = false;

    for (int i = 1; i < static_cast<int>(tokens.size()); i += 2) {
        std::string type = tokens[i];

        uint64_t value = 0;

        if (static_cast<int>(tokens.size()) > i + 1) value = std::stoi(tokens[i + 1]);

        if (type == "depth") d = static_cast<PLY_TYPE>(value);

        else if (type == "nodes") mcts_engine->max_nodes = value;

        else if (type == "perft") perft_depth = static_cast<PLY_TYPE>(value);

        else if (type == "movetime") movetime = static_cast<double>(value);

        else if (type == "wtime") wtime = static_cast<double>(value);
        else if (type == "btime") btime = static_cast<double>(value);

        else if (type == "winc") winc = static_cast<double>(value);
        else if (type == "binc") binc = static_cast<double>(value);

        else if (type == "movestogo") movestogo = static_cast<long>(value);
        else if (type == "infinite") infinite = true;

    }

    if (perft_depth > 0) {
        uci_perft(position, perft_depth, 0);
        return;
    }
    if (infinite || (d && tokens.size() == 3)) {
        mcts_engine->max_time = TIME_INF;
    }
    else {
        double self_time = (position.side == 0) ? wtime : btime;
        double inc = (position.side == 0) ? winc : binc;

        time_handler(std::max<double>(self_time - 30, 0.0), inc, movetime, movestogo);
    }

    mcts_engine->search();

    //iterative_search(engine, position);
}


void UCI::uci_loop() {

    std::cout << std::string(ENGINE_NAME) + " " + std::string(ENGINE_VERSION) + " by " + std::string(ENGINE_AUTHOR) << std::endl;
    while (getline(std::cin, msg)) {
        tokens.clear();

        tokens = split(msg, ' ');

        if (msg == "quit") {
            break;
        }

        if (msg == "stop") {

        }

        else if (msg == "uci") {
            std::cout << "id name " + std::string(ENGINE_NAME) + " " + std::string(ENGINE_VERSION) << std::endl;
            std::cout << "id author Alexander Tian" << std::endl;

            std::cout << "uciok" << std::endl;
        }

        else if (tokens[0] == "setoption" && tokens.size() >= 5) {

        }

        else if (msg == "isready") {
            std::cout << "readyok" << std::endl;
        }

        else if (msg == "ucinewgame") {
            current_move_idx = 0;
            Position& position = mcts_engine->position;
            position.set_fen(START_FEN);
            mcts_engine->new_game();
        }

        else if (tokens[0] == "position") {
            parse_position();
        }

        else if (tokens[0] == "go") {
            parse_go();
        }

        else if (tokens[0] == "bench") {
            run_bench(*mcts_engine, BENCH_DEPTH);
        }
    }
}