// Use Metall w/ single thread
#define METALL_DISABLE_CONCURRENCY

#pragma once

#include "hdf5_tools.hpp"
#include "type_info.hpp"
#include <binsparse/containers/matrices.hpp>
#include <binsparse/detail.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <type_traits>
#include <metall/metall.hpp>
#include <metall/container/string.hpp>

#include <binsparse/c_bindings/allocator_wrapper.hpp>
#include <binsparse/matrix_market/matrix_market.hpp>

namespace binsparse {

inline const std::string version = "0.1";

template <typename T>
void write_dense_vector(H5::Group& f, std::span<T> v,
                        nlohmann::json user_keys = {}) {
  hdf5_tools::write_dataset(f, "values", v);

  using json = nlohmann::json;
  json j;
  j["binsparse"]["version"] = version;
  j["binsparse"]["format"] = "DVEC";
  j["binsparse"]["shape"] = {v.size()};
  j["binsparse"]["number_of_stored_values"] = v.size();
  j["binsparse"]["data_types"]["values"] = type_info<T>::label();

  for (auto&& v : user_keys.items()) {
    j[v.key()] = v.value();
  }

  hdf5_tools::set_attribute(f, "binsparse", j.dump(2));
}

template <typename T, typename Allocator = std::allocator<T>>
auto read_dense_vector(std::string fname, Allocator&& alloc = Allocator{}) {
  H5::H5File f(fname.c_str(), H5F_ACC_RDWR);

  auto metadata = hdf5_tools::get_attribute(f, "binsparse");

  using json = nlohmann::json;
  auto data = json::parse(metadata);

  auto binsparse_metadata = data["binsparse"];

  auto format = __detail::unalias_format(binsparse_metadata["format"]);

  assert(format == "DVEC");

  auto nvalues = binsparse_metadata["shape"][0];
  auto nnz = binsparse_metadata["number_of_stored_values"];

  assert(nvalues == nnz);

  auto values = hdf5_tools::read_dataset<T>(f, "values", alloc);

  assert(values.size() == nvalues);

  return values;
}

// Dense Format

template <typename T, typename I, typename Order>
void write_dense_matrix(H5::Group& f, dense_matrix<T, I, Order> m,
                        nlohmann::json user_keys = {}) {
  std::span<T> values(m.values, m.m * m.n);

  hdf5_tools::write_dataset(f, "values", values);

  using json = nlohmann::json;
  json j;
  j["binsparse"]["version"] = version;
  j["binsparse"]["format"] = __detail::get_matrix_format_string(m);
  j["binsparse"]["shape"] = {m.m, m.n};
  j["binsparse"]["number_of_stored_values"] = m.m * m.n;

  if (!m.is_iso) {
    j["binsparse"]["data_types"]["values"] =
        std::string("iso[") + type_info<T>::label() + "]";
  } else {
    j["binsparse"]["data_types"]["values"] = type_info<T>::label();
  }

  if (m.structure != general) {
    j["binsparse"]["structure"] =
        __detail::get_structure_name(m.structure).value();
  }

  for (auto&& v : user_keys.items()) {
    j[v.key()] = v.value();
  }

  hdf5_tools::set_attribute(f, "binsparse", j.dump(2));
}

template <typename T, typename I, typename Order>
void write_dense_matrix(std::string fname, dense_matrix<T, I, Order> m,
                        nlohmann::json user_keys = {}) {
  H5::H5File f(fname.c_str(), H5F_ACC_TRUNC);
  write_dense_matrix(f, m, user_keys);
  f.close();
}

template <typename T, typename I, typename Order,
          typename Allocator = std::allocator<T>>
auto read_dense_matrix(std::string fname, Allocator&& alloc = Allocator{}) {
  H5::H5File f(fname.c_str(), H5F_ACC_RDWR);

  auto metadata = hdf5_tools::get_attribute(f, "binsparse");

  using json = nlohmann::json;
  auto data = json::parse(metadata);

  auto binsparse_metadata = data["binsparse"];

  auto format = __detail::unalias_format(binsparse_metadata["format"]);

  assert(format ==
         __detail::get_matrix_format_string(dense_matrix<T, I, Order>{}));

  auto nrows = binsparse_metadata["shape"][0];
  auto ncols = binsparse_metadata["shape"][1];
  auto nnz = binsparse_metadata["number_of_stored_values"];

  bool is_iso = false;
  if (std::string(binsparse_metadata["data_types"]["values"])
          .starts_with("iso")) {
    is_iso = true;
  }

  auto values = hdf5_tools::read_dataset<T>(f, "values", alloc);

  structure_t structure = general;

  if (binsparse_metadata.contains("structure")) {
    structure = __detail::parse_structure(binsparse_metadata["structure"]);
  }

  return dense_matrix<T, I, Order>{values.data(), nrows, ncols, structure,
                                   is_iso};
}

// CSR Format

template <typename T, typename I>
nlohmann::json make_csr_json_metadata(csr_matrix<T, I> m, nlohmann::json user_keys) {
  using json = nlohmann::json;
  json j;
  j["binsparse"]["version"] = version;
  j["binsparse"]["format"] = "CSR";
  j["binsparse"]["shape"] = {m.m, m.n};
  j["binsparse"]["number_of_stored_values"] = m.nnz;
  j["binsparse"]["data_types"]["pointers_to_1"] = type_info<I>::label();
  j["binsparse"]["data_types"]["indices_1"] = type_info<I>::label();

  if (!m.is_iso) {
    j["binsparse"]["data_types"]["values"] =
        std::string("iso[") + type_info<T>::label() + "]";
  } else {
    j["binsparse"]["data_types"]["values"] = type_info<T>::label();
  }

  if (m.structure != general) {
    j["binsparse"]["structure"] =
        __detail::get_structure_name(m.structure).value();
  }

  for (auto&& v : user_keys.items()) {
    j[v.key()] = v.value();
  }

  return j;
}

template <typename T, typename I>
void write_csr_matrix(H5::Group& f, csr_matrix<T, I> m,
                      nlohmann::json user_keys = {},
                      const int def_level = 9) {
  std::span<T> values(m.values, m.nnz);
  std::span<I> colind(m.colind, m.nnz);
  std::span<I> row_ptr(m.row_ptr, m.m + 1);

  hdf5_tools::write_dataset(f, "values", values, def_level);
  hdf5_tools::write_dataset(f, "indices_1", colind,def_level);
  hdf5_tools::write_dataset(f, "pointers_to_1", row_ptr,def_level);

  auto j = make_csr_json_metadata(m, user_keys);
  hdf5_tools::set_attribute(f, "binsparse", j.dump(2));
}

template <typename T, typename I>
void write_csr_matrix(std::string fname, csr_matrix<T, I> m,
                      nlohmann::json user_keys = {},
                      const int def_level = 9) {
  H5::H5File f(fname.c_str(), H5F_ACC_TRUNC);
  write_csr_matrix(f, m, user_keys, def_level);
  f.close();
}

template <typename T, typename I>
void write_csr_matrix(metall::manager& manager, csr_matrix<T, I> m,
                      nlohmann::json user_keys = {}) {
  // We do nothing with 'm' as it is already in metall
  // Just write the metadata
  auto j = make_csr_json_metadata(m, user_keys);
  // Write the metadata as a string to Metall
  manager.construct<metall::container::string>("binsparse-metadata")(j.dump(2), manager.get_allocator<>());
}

inline void parse_csr_json_metadata(const nlohmann::json& data, std::size_t& nrows,
                                    std::size_t& ncols, std::size_t& nnz, bool& is_iso,
                                    structure_t& structure) {
  auto binsparse_metadata = data["binsparse"];

  assert(binsparse_metadata["format"] == "CSR");

  nrows = binsparse_metadata["shape"][0];
  ncols = binsparse_metadata["shape"][1];
  nnz = binsparse_metadata["number_of_stored_values"];

  is_iso = false;
  if (std::string(binsparse_metadata["data_types"]["values"])
          .starts_with("iso")) {
    is_iso = true;
  }

  structure = general;
  if (binsparse_metadata.contains("structure")) {
    structure = __detail::parse_structure(binsparse_metadata["structure"]);
  }
}

template <typename T, typename I, typename Allocator>
csr_matrix<T, I> read_csr_matrix(std::string fname, Allocator&& alloc) {
  H5::H5File f(fname.c_str(), H5F_ACC_RDWR);

  auto metadata = hdf5_tools::get_attribute(f, "binsparse");

  using json = nlohmann::json;
  auto data = json::parse(metadata);

  std::size_t nrows, ncols, nnz;
  bool is_iso;
  structure_t structure;
  parse_csr_json_metadata(data, nrows, ncols, nnz, is_iso, structure);

  typename std::allocator_traits<
      std::remove_cvref_t<Allocator>>::template rebind_alloc<I>
      i_alloc(alloc);

  auto values = hdf5_tools::read_dataset<T>(f, "values", alloc);
  auto colind = hdf5_tools::read_dataset<I>(f, "indices_1", i_alloc);
  auto row_ptr = hdf5_tools::read_dataset<I>(f, "pointers_to_1", i_alloc);

  return csr_matrix<T, I>{values.data(), colind.data(), row_ptr.data(), nrows,
                          ncols,         nnz,           structure,      is_iso};
}

template <typename T, typename I>
csr_matrix<T, I> read_csr_matrix(std::string fname) {
  return read_csr_matrix<T, I>(fname, std::allocator<T>{});
}

template <typename T, typename I>
csr_matrix<T, I> read_csr_matrix(metall::manager& manager) {
  // Find the metadata in Metall datastore
  // This process just looks up the address of the instance associated with the key
  // and returns a pointer to the instance.
  // No data is copied.
  auto& metadata = *(manager.find<metall::container::string>("binsparse-metadata").first);

  using json = nlohmann::json;
  auto data = json::parse(metadata);

  std::size_t nrows, ncols, nnz;
  bool is_iso;
  structure_t structure;
  parse_csr_json_metadata(data, nrows, ncols, nnz, is_iso, structure);

  using A = metall::manager::allocator_type<T>;
  using M = binsparse::__detail::csr_matrix_owning<T, I, A>;

  // Find the CSR matrix in Metall datastore
  auto x = manager.find<M>("binsparse-matrix").first;
  assert(x);

  return csr_matrix<T, I>{x->values().data(), x->colind().data(), x->rowptr().data(), nrows,
                          ncols,         nnz,           structure,      is_iso};
}

// CSC Format

template <typename T, typename I>
nlohmann::json make_csc_json_metadata(csc_matrix<T, I> m, nlohmann::json user_keys) {
  using json = nlohmann::json;
  json j;
  j["binsparse"]["version"] = version;
  j["binsparse"]["format"] = "CSR";
  j["binsparse"]["shape"] = {m.m, m.n};
  j["binsparse"]["number_of_stored_values"] = m.nnz;
  j["binsparse"]["data_types"]["pointers_to_1"] = type_info<I>::label();
  j["binsparse"]["data_types"]["indices_1"] = type_info<I>::label();

  if (!m.is_iso) {
    j["binsparse"]["data_types"]["values"] =
        std::string("iso[") + type_info<T>::label() + "]";
  } else {
    j["binsparse"]["data_types"]["values"] = type_info<T>::label();
  }

  if (m.structure != general) {
    j["binsparse"]["structure"] =
        __detail::get_structure_name(m.structure).value();
  }

  for (auto&& v : user_keys.items()) {
    j[v.key()] = v.value();
  }
  return j;
}

template <typename T, typename I>
void write_csc_matrix(H5::Group& f, csc_matrix<T, I> m,
                      nlohmann::json user_keys = {}) {
  std::span<T> values(m.values, m.nnz);
  std::span<I> rowind(m.rowind, m.nnz);
  std::span<I> col_ptr(m.col_ptr, m.m + 1);

  hdf5_tools::write_dataset(f, "values", values);
  hdf5_tools::write_dataset(f, "indices_1", rowind);
  hdf5_tools::write_dataset(f, "pointers_to_1", col_ptr);

  auto j = make_csc_json_metadata(m, user_keys);

  hdf5_tools::set_attribute(f, "binsparse", j.dump(2));
}

template <typename T, typename I>
void write_csc_matrix(std::string fname, csc_matrix<T, I> m,
                      nlohmann::json user_keys = {}) {
  H5::H5File f(fname.c_str(), H5F_ACC_TRUNC);
  write_csc_matrix(f, m, user_keys);
  f.close();
}

inline void parse_csc_json_metadata(const nlohmann::json& data, std::size_t& nrows,
                                    std::size_t& ncols, std::size_t& nnz, bool& is_iso,
                                    structure_t& structure) {
  auto binsparse_metadata = data["binsparse"];

  assert(binsparse_metadata["format"] == "CSC");

  nrows = binsparse_metadata["shape"][0];
  ncols = binsparse_metadata["shape"][1];
  nnz = binsparse_metadata["number_of_stored_values"];

  is_iso = false;
  if (std::string(binsparse_metadata["data_types"]["values"])
          .starts_with("iso")) {
    is_iso = true;
  }

  structure = general;
  if (binsparse_metadata.contains("structure")) {
    structure = __detail::parse_structure(binsparse_metadata["structure"]);
  }
}

template <typename T, typename I, typename Allocator>
csc_matrix<T, I> read_csc_matrix(std::string fname, Allocator&& alloc) {
  H5::H5File f(fname.c_str(), H5F_ACC_RDWR);

  auto metadata = hdf5_tools::get_attribute(f, "binsparse");

  using json = nlohmann::json;
  auto data = json::parse(metadata);

  std::size_t nrows, ncols, nnz;
  bool is_iso;
  structure_t structure;
  parse_csc_json_metadata(data, nrows, ncols, nnz, is_iso, structure);

  typename std::allocator_traits<
      std::remove_cvref_t<Allocator>>::template rebind_alloc<I>
      i_alloc(alloc);

  auto values = hdf5_tools::read_dataset<T>(f, "values", alloc);
  auto rowind = hdf5_tools::read_dataset<I>(f, "indices_1", i_alloc);
  auto col_ptr = hdf5_tools::read_dataset<I>(f, "pointers_to_1", i_alloc);

  return csc_matrix<T, I>{values.data(), rowind.data(), col_ptr.data(), nrows,
                          ncols,         nnz,           structure,      is_iso};
}

template <typename T, typename I>
csc_matrix<T, I> read_csc_matrix(std::string fname) {
  return read_csc_matrix<T, I>(fname, std::allocator<T>{});
}

// COO Format

template <typename T, typename I>
nlohmann::json make_coo_json_metadata(coo_matrix<T, I> m, nlohmann::json user_keys) {
  using json = nlohmann::json;
  json j;
  j["binsparse"]["version"] = version;
  j["binsparse"]["format"] = "COO";
  j["binsparse"]["shape"] = {m.m, m.n};
  j["binsparse"]["number_of_stored_values"] = m.nnz;
  j["binsparse"]["data_types"]["indices_0"] = type_info<I>::label();
  j["binsparse"]["data_types"]["indices_1"] = type_info<I>::label();

  if (!m.is_iso) {
    j["binsparse"]["data_types"]["values"] =
        std::string("iso[") + type_info<T>::label() + "]";
  } else {
    j["binsparse"]["data_types"]["values"] = type_info<T>::label();
  }

  if (m.structure != general) {
    j["binsparse"]["structure"] =
        __detail::get_structure_name(m.structure).value();
  }

  for (auto&& v : user_keys.items()) {
    j[v.key()] = v.value();
  }
  return j;
}

template <typename T, typename I>
void write_coo_matrix(H5::Group& f, coo_matrix<T, I> m,
                      nlohmann::json user_keys = {}) {
  std::span<T> values(m.values, m.nnz);
  std::span<I> rowind(m.rowind, m.nnz);
  std::span<I> colind(m.colind, m.nnz);

  hdf5_tools::write_dataset(f, "values", values);
  hdf5_tools::write_dataset(f, "indices_0", rowind);
  hdf5_tools::write_dataset(f, "indices_1", colind);

  auto j = make_coo_json_metadata(m, user_keys);

  hdf5_tools::set_attribute(f, "binsparse", j.dump(2));
}

template <typename T, typename I>
void write_coo_matrix(std::string fname, coo_matrix<T, I> m,
                      nlohmann::json user_keys = {}) {
  H5::H5File f(fname.c_str(), H5F_ACC_TRUNC);
  write_coo_matrix(f, m, user_keys);
  f.close();
}

template <typename T, typename I>
void write_coo_matrix(metall::manager& manager, coo_matrix<T, I> m,
                      nlohmann::json user_keys = {}) {

  auto j = make_coo_json_metadata(m, user_keys);
  manager.construct<metall::container::string>("binsparse-metadata")(j.dump(2), manager.get_allocator<>());
}


inline void parse_coo_json_metadata(const nlohmann::json& data, std::size_t& nrows,
                                    std::size_t& ncols, std::size_t& nnz, bool& is_iso,
                                    structure_t& structure) {
  auto binsparse_metadata = data["binsparse"];

  auto format = __detail::unalias_format(binsparse_metadata["format"]);
  assert(format == "COOR" || format == "COOC");

  nrows = binsparse_metadata["shape"][0];
  ncols = binsparse_metadata["shape"][1];
  nnz = binsparse_metadata["number_of_stored_values"];

  is_iso = false;
  if (std::string(binsparse_metadata["data_types"]["values"])
          .starts_with("iso")) {
    is_iso = true;
  }

  structure = general;
  if (binsparse_metadata.contains("structure")) {
    structure = __detail::parse_structure(binsparse_metadata["structure"]);
  }
}

template <typename T, typename I, typename Allocator>
coo_matrix<T, I> read_coo_matrix(std::string fname, Allocator&& alloc) {
  H5::H5File f(fname.c_str(), H5F_ACC_RDWR);

  auto metadata = hdf5_tools::get_attribute(f, "binsparse");

  using json = nlohmann::json;
  auto data = json::parse(metadata);

  std::size_t nrows, ncols, nnz;
  bool is_iso;
  structure_t structure;
  parse_coo_json_metadata(data, nrows, ncols, nnz, is_iso, structure);

  typename std::allocator_traits<
      std::remove_cvref_t<Allocator>>::template rebind_alloc<I>
      i_alloc(alloc);

  auto values = hdf5_tools::read_dataset<T>(f, "values", alloc);
  auto rows = hdf5_tools::read_dataset<I>(f, "indices_0", i_alloc);
  auto cols = hdf5_tools::read_dataset<I>(f, "indices_1", i_alloc);

  return coo_matrix<T, I>{values.data(), rows.data(), cols.data(), nrows,
                          ncols,         nnz,         structure,   is_iso};
}

template <typename T, typename I>
coo_matrix<T, I> read_coo_matrix(std::string fname) {
  return read_coo_matrix<T, I>(fname, std::allocator<T>{});
}

template <typename T, typename I>
coo_matrix<T, I> read_coo_matrix(metall::manager& manager) {
  auto& metadata = *(manager.find<metall::container::string>("binsparse-metadata").first);

  using json = nlohmann::json;
  auto data = json::parse(metadata);

  std::size_t nrows, ncols, nnz;
  bool is_iso;
  structure_t structure;
  parse_coo_json_metadata(data, nrows, ncols, nnz, is_iso, structure);

  using A = metall::manager::allocator_type<T>;
  using M = binsparse::__detail::coo_matrix_owning<T, I, A>;
  auto x = manager.find<M>("binsparse-matrix").first;
  assert(x);

  return coo_matrix<T, I>{x->values().data(), x->rowind().data(), x->colind().data(), nrows,
                          ncols,         nnz,         structure,   is_iso};
}

inline auto inspect(std::string fname) {
  H5::H5File f(fname.c_str(), H5F_ACC_RDWR);

  auto metadata = hdf5_tools::get_attribute(f, "binsparse");

  using json = nlohmann::json;
  auto data = json::parse(metadata);

  auto binsparse_metadata = data["binsparse"];

  assert(binsparse_metadata["version"] >= 0.1);

  return data;
}

} // namespace binsparse