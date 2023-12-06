#ifndef RESULT_COLLECTOR
#define RESULT_COLLECTOR

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <vector>


namespace rc {

template <typename value_type>
class result_column {
    std::string name_;
    const value_type* value_;

    friend class result_collector;
public:
    result_column(const std::string& name, const value_type* value) : name_{name}, value_{value} {}

    template <typename elem_type>
    result_column<elem_type> operator<<(const elem_type& value) {
        return result_column<elem_type>(name_, &value);
    }
};


inline
result_column<std::string> col(const std::string& name) {
    return result_column<std::string>(name, nullptr);
}


class result_collector {
    struct column {
        std::string name;
        size_t width;
        std::vector<std::string> values;
    };

    std::stringstream buf;
    std::string sep;
    std::string empty;
    std::vector<column> columns;
    std::map<std::string, std::string> current_row;
    size_t num_rows = 0;
    size_t precision = 3;
    bool header;

    std::string padding(const column& col, size_t entry_size) {
        return std::string(col.width - entry_size, ' ');
    }

public:
    explicit result_collector(const std::string& empty = "", const std::string& sep = ", ", bool header = true, bool verbose = false) : sep{sep}, empty{empty}, header{header} {
        buf << std::fixed << std::setprecision(precision);
    }
    result_collector(const result_collector&) = delete;
    result_collector& operator=(const result_collector&) = delete;
    result_collector(result_collector&&) = delete;
    result_collector& operator=(result_collector&&) = delete;

    void reset_line() {
        current_row.clear();
    }

    void commit_line() {
        /*
        // temporary fix: also print to stdout
        {
            bool first = true;
            for (const auto& new_entry : current_row) {
                const auto& col_name = new_entry.first;
                const auto& value = new_entry.second;
                if (!first) std::cout << sep;
                first = false;
                std::cout << col_name << "=" << value;
            }
            std::cout << std::endl;
        }
        */
        // existing columns
        for (auto& col : columns) {
            std::map<std::string, std::string>::iterator it;
            if ((it = current_row.find(col.name)) != current_row.end()) {
                const auto& value = it->second;

                col.values.push_back(value);
                col.width = std::max(col.width, value.size());

                current_row.erase(it);
            } else {
                col.values.push_back(empty);
                col.width = std::max(col.width, empty.size());
            }
        }
        // new columns
        for (const auto& new_entry : current_row) {
            const auto& col_name = new_entry.first;
            const auto& value = new_entry.second;

            column new_column {col_name, std::max(col_name.size(), value.size()), std::vector<std::string>(num_rows, empty)};
            new_column.values.push_back(value);

            columns.push_back(std::move(new_column));
        }
        current_row.clear();
        num_rows += 1;
    }

    void write_csv(std::basic_ostream<char>& stream) {
        bool first = true;
        if (header) {
            for (const auto& col : columns) {
                if (!first) stream << sep;
                stream << padding(col, col.name.size()) << col.name;
                first = false;
            }
            stream << "\n";
        }
        for (size_t i = 0; i < num_rows; ++i) {
            first = true;
            for (const auto& col : columns) {
                const auto& value = col.values[i];
                if (!first) stream << sep;
                stream << padding(col, value.size()) << value;
                first = false;
            }
            stream << "\n";
        }
    }

    template <typename value_type>
    result_collector& operator|(const result_column<value_type>& rc) {
        if (rc.value_ == nullptr) {
            current_row[rc.name_] = empty;
        } else {
            buf.str("");
            buf << *rc.value_;
            current_row[rc.name_] = buf.str();
        }
        return *this;
    }

    template <typename value_type>
    result_collector& add(const std::string& name, const value_type& value) {
      buf.str("");
      buf << value;
      current_row[name] = buf.str();
      return *this;
    }
};

}

#endif
