"""
02_attributes.py
Stage 2 - Sensitive attributes for the fairness audit (not all are model features).

 * region (7 island groups) from province
 * underdeveloped district (3T) flag: Presidential Regulation (Perpres) 63/2020,
   62 regencies designated "daerah tertinggal" for 2020-2024 (covers SKI 2023 fieldwork)
 * poverty proxy 1: household holds subsidised national health insurance (BPJS PBI),
   whose beneficiaries are drawn from the national social-welfare registry (DTKS)
 * poverty proxy 2: housing/WASH deprivation index (first principal component of
   binary housing & WASH indicators + log floor area per person), weighted quintiles
 * urban/rural, age band, sex

Outputs: data/attributes.pkl, results/T_attr_3T_mapping.csv, results/T_attr_profile.csv
"""
import numpy as np
import pandas as pd
from common import *

A = pd.read_pickle(ANALYTIC)
df = A["df"].reset_index(drop=True)
raw = pd.read_pickle(RAW_CACHE).drop_duplicates("ID ART").set_index("ID ART")
R = raw.loc[df["ID ART"].values].reset_index()

REGION = {
    "Sumatera": ["Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Jambi", "Sumatera Selatan",
                 "Bengkulu", "Lampung", "Kep.Bangka Belitung", "Kepulauan Riau"],
    "Java": ["DKI Jakarta", "Jawa Barat", "Jawa Tengah", "DI Yogyakarta", "Jawa Timur", "Banten"],
    "Bali-Nusa Tenggara": ["Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur"],
    "Kalimantan": ["Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur",
                   "Kalimantan Utara"],
    "Sulawesi": ["Sulawesi Utara", "Sulawesi Tengah", "Sulawesi Selatan", "Sulawesi Tenggara",
                 "Gorontalo", "Sulawesi Barat"],
    "Maluku": ["Maluku", "Maluku Utara"],
    "Papua": ["Papua", "Papua Barat", "Papua Selatan", "Papua Tengah", "Papua Pegunungan",
              "Papua Barat Daya"],
}
p2r = {p: r for r, ps in REGION.items() for p in ps}
out = pd.DataFrame({"ID ART": df["ID ART"].values})
out["province"] = df["province"].values
miss = set(out.province) - set(p2r)
assert not miss, miss
out["region"] = out.province.map(p2r)
out["east"] = out.region.isin(["Bali-Nusa Tenggara", "Maluku", "Papua"]).astype(int)
out["district"] = R["2. Kabupaten/Kota"].astype(str).values

# ---------------------------------------------------------------- 3T (Perpres 63/2020)
PERPRES_63_2020 = {
    "Sumatera Utara": ["Nias", "Nias Selatan", "Nias Utara", "Nias Barat"],
    "Sumatera Barat": ["Kepulauan Mentawai"],
    "Sumatera Selatan": ["Musi Rawas Utara"],
    "Lampung": ["Pesisir Barat"],
    "Nusa Tenggara Barat": ["Lombok Utara"],
    "Nusa Tenggara Timur": ["Sumba Barat", "Sumba Timur", "Kupang", "Timor Tengah Selatan", "Belu",
                            "Alor", "Lembata", "Rote Ndao", "Sumba Tengah", "Sumba Barat Daya",
                            "Manggarai Timur", "Sabu Raijua", "Malaka"],
    "Sulawesi Tengah": ["Donggala", "Tojo Una-Una", "Sigi"],
    "Maluku": ["Maluku Tenggara Barat", "Kepulauan Aru", "Seram Bagian Barat", "Seram Bagian Timur",
               "Maluku Barat Daya", "Buru Selatan"],
    "Maluku Utara": ["Kepulauan Sula", "Pulau Taliabu"],
    "Papua Barat": ["Teluk Wondama", "Teluk Bintuni", "Sorong Selatan", "Sorong", "Tambrauw",
                    "Maybrat", "Manokwari Selatan", "Pegunungan Arfak"],
    "Papua": ["Jayawijaya", "Nabire", "Paniai", "Puncak Jaya", "Boven Digoel", "Mappi", "Asmat",
              "Yahukimo", "Pegunungan Bintang", "Tolikara", "Keerom", "Waropen", "Supiori",
              "Mamberamo Raya", "Nduga", "Lanny Jaya", "Mamberamo Tengah", "Yalimo", "Puncak",
              "Dogiyai", "Intan Jaya", "Deiyai"],
}
# name in SKI file when it differs from the regulation (renamed regency / spelling in SKI)
ALIAS = {"Maluku Tenggara Barat": "Kepulauan Tanimbar",   # renamed 2019
         "Mamberamo Tengah": "Mamberano Tengah"}            # spelling in SKI file
n_list = sum(len(v) for v in PERPRES_63_2020.values())
assert n_list == 62, n_list
skinames = set(out.district)
maprows, t3names = [], set()
for prov, regs in PERPRES_63_2020.items():
    for r in regs:
        s = ALIAS.get(r, r)
        found = s in skinames
        n = int((out.district == s).sum())
        prov_now = sorted(out.loc[out.district == s, "province"].unique())
        maprows.append(dict(province_perpres=prov, regency_perpres=r, name_in_ski=s,
                            matched=found, province_in_ski_2023="; ".join(prov_now), n_children=n))
        if found:
            t3names.add(s)
pd.DataFrame(maprows).to_csv(os.path.join(RES_DIR, "T_attr_3T_mapping.csv"), index=False)
out["t3"] = out.district.isin(t3names).astype(int)
print("3T regencies matched:", sum(r["matched"] for r in maprows), "/ 62; children", out.t3.sum())

# ---------------------------------------------------------------- poverty proxies
jkn = R["11. Kepemilikan Jaminan Kesehatan"].astype(str).values
out["pbi"] = (jkn == "BPJS PBI").astype(int)
out["insurance"] = np.select([jkn == "BPJS PBI", np.isin(jkn, ["BPJS NON PBI", "ASURANSI SWASTA",
                              "BPJS NON PBI DAN ASURANSI SWASTA", "KOMBINASI LAIN", "JAMKESDA", "LAINNYA"]),
                              jkn == "TIDAK ADA"], ["PBI (subsidised)", "Other insurance", "Uninsured"], "Other insurance")

roof = R["25.Jenis bahan bangunan utama atap rumah terluas?"]
floor = R["27.Jenis bahan bangunan utama lantai rumah terluas?"]
wall = R["28.Apakah bahan bangunan utama dinding rumah terluas?"]
water = R["1.Apakah jenis sarana air yang UTAMA digunakan oleh [RUMAH TANGGA] untuk keperluan minum saat ini?"]
toilet = R["11.Apakah [RUMAH TANGGA] memiliki fasilitas tempat buang air besar dan siapa saja yang menggunakan?"]
sink = R["14.Dimanakah tinja tersebut dialirkan?"]
area = pd.to_numeric(R["2.Luas lantai bangunan rumah  (BULATKAN DALAM METER PERSEGI)"], errors="coerce")
hh = pd.to_numeric(R["1. Banyaknya anggota rumah tangga:"], errors="coerce").clip(lower=1)
ind = pd.DataFrame({
    "finished_roof": roof.isin(["Seng", "Tanah Liat/Genteng", "Asbes", "Beton, aspal"]),
    "finished_floor": floor.isin(["Keramk", "Ubin/tegel/teraso", "Marmer/Granit", "Parket/vinil/Karpet"]),
    "masonry_wall": wall.eq("Tembok"),
    "improved_water": ~water.isin(["Sumur gali tak terlindung", "Mata air tidak terlindung",
                                   "Air permukaan (sungai/danau/irigasi)"]),
    "private_toilet": toilet.eq("Ada, digunakan hanya ART sendiri"),
    "safe_excreta": sink.isin(["Septic tank", "IPAL"]),
}).astype(float)
ind["log_area_pp"] = np.log1p((area / hh).clip(upper=200)).fillna(np.log1p((area / hh).median()))
Z = (ind - ind.mean()) / ind.std()
_, _, vt = np.linalg.svd(Z.values, full_matrices=False)
pc1 = Z.values @ vt[0]
if np.corrcoef(pc1, ind["finished_floor"])[0, 1] < 0:
    pc1 = -pc1
w = df[W].values
order = np.argsort(pc1); cw = np.cumsum(w[order]) / w.sum()
q = np.empty(len(pc1), int); q[order] = np.minimum((cw * 5).astype(int), 4) + 1
out["housing_q"] = q            # 1 = most deprived
out["housing_score"] = pc1
pd.DataFrame({"indicator": ind.columns, "loading_pc1": vt[0] * np.sign(np.corrcoef(Z.values @ vt[0], ind["finished_floor"])[0, 1]),
              "weighted_mean": [np.average(ind[c], weights=w) for c in ind.columns]}).round(3) \
  .to_csv(os.path.join(RES_DIR, "T_attr_housing_index.csv"), index=False)

out["urban"] = df["urban"].values
out["age_band"] = pd.cut(df["age_months"], [-0.1, 6, 24, 61], right=False,
                         labels=["0-5 mo", "6-23 mo", "24-59 mo"]).astype(str).values
out["sex"] = df["sex"].map({1: "Boys", 2: "Girls"}).values
out["stunted"] = df["stunted"].values
out[W] = w; out[PSU] = df[PSU].values; out[STRATA] = df[STRATA].values
out.to_pickle(os.path.join(DATA_DIR, "attributes.pkl"))

# ---------------------------------------------------------------- profile table
rows = []
def add(attr, lvl, m):
    rows.append(dict(attribute=attr, group=lvl, n=int(m.sum()), weighted_pct=100 * w[m].sum() / w.sum(),
                     stunting_w_pct=100 * np.average(out.stunted[m], weights=w[m])))
add("All", "All", np.ones(len(out), bool))
for r in REGION:
    add("Region", r, (out.region == r).values)
add("Underdeveloped district (3T)", "3T regency", out.t3.values == 1)
add("Underdeveloped district (3T)", "Non-3T", out.t3.values == 0)
add("Residence", "Urban", out.urban.values == 1); add("Residence", "Rural", out.urban.values == 0)
for lvl in ["PBI (subsidised)", "Other insurance", "Uninsured"]:
    add("Health insurance (poverty proxy)", lvl, (out.insurance == lvl).values)
for k in range(1, 6):
    add("Housing deprivation quintile", f"Q{k}" + (" (most deprived)" if k == 1 else " (least deprived)" if k == 5 else ""),
        out.housing_q.values == k)
P = pd.DataFrame(rows).round(2)
P.to_csv(os.path.join(RES_DIR, "T_attr_profile.csv"), index=False)
print(P.to_string())
print("corr PBI vs housing score:", np.corrcoef(out.pbi, out.housing_score)[0, 1].round(3))
