export const HCPCS_BUCKET_NAMES: Record<number, string> = {
  0: 'Anesthesia',
  1: 'Surgery',
  2: 'Radiology',
  3: 'Lab/Pathology',
  4: 'Medicine/E&M',
  5: 'HCPCS Level II',
};

export const CENSUS_REGION_NAMES: Record<number, string> = {
  1: 'Northeast',
  2: 'Midwest',
  3: 'South',
  4: 'West',
};

export const AGE_GROUP_LABELS: Record<number, string> = {
  1: 'Under 65',
  2: '65-74',
  3: '75+',
};

export const INCOME_LABELS: Record<number, string> = {
  1: 'Below Median',
  2: 'Above Median',
};

// Maps state abbreviation to census region integer
// Source: generate_synthetic_mcbs.py CENSUS_REGIONS + REGION_NAME_TO_INT
export const STATE_TO_REGION: Record<string, number> = {
  CT: 1, ME: 1, MA: 1, NH: 1, RI: 1, VT: 1, NJ: 1, NY: 1, PA: 1,
  DE: 3, FL: 3, GA: 3, MD: 3, NC: 3, SC: 3, VA: 3, DC: 3, WV: 3,
  AL: 3, KY: 3, MS: 3, TN: 3, AR: 3, LA: 3, OK: 3, TX: 3,
  IL: 2, IN: 2, MI: 2, OH: 2, WI: 2, IA: 2, KS: 2, MN: 2, MO: 2,
  NE: 2, ND: 2, SD: 2,
  AZ: 4, CO: 4, ID: 4, MT: 4, NV: 4, NM: 4, UT: 4, WY: 4,
  AK: 4, CA: 4, HI: 4, OR: 4, WA: 4,
};

export const FEATURE_DISPLAY_NAMES: Record<string, string> = {
  Avg_Sbmtd_Chrg: 'Submitted Charge',
  HCPCS_Cd_idx: 'Procedure Code',
  hcpcs_bucket: 'Service Category',
  Rndrng_Prvdr_Type_idx: 'Provider Specialty',
  srvcs_per_bene: 'Services per Beneficiary',
  log_srvcs: 'Log(Service Count)',
  log_benes: 'Log(Beneficiary Count)',
  Bene_Avg_Risk_Scre: 'Beneficiary Risk Score',
  Rndrng_Prvdr_State_Abrvtn_idx: 'State',
  place_of_srvc_flag: 'Place of Service',
};

export const NAV_LINKS = [
  { label: 'Estimator', href: '/' },
  { label: 'Forecast', href: '/forecast' },
  { label: 'Investigations', href: '/investigations' },
  { label: 'About', href: '/about' },
];
