from src.matching.matcher import SecurityMatcher, SynCompanyMatcher, SynSecurityMatcher, RealCompanyMatcher, WDCMatcher

matchers_dict = {
    'synthetic_companies': SynCompanyMatcher,
    'synthetic_securities': SynSecurityMatcher,
    'real_securities': SecurityMatcher,
    'real_companies': RealCompanyMatcher,
    'wdc': WDCMatcher,
}
