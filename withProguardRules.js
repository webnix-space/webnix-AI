const { withAppBuildGradle } = require('@expo/config-plugins');

module.exports = function withProguardRules(config) {
  return withAppBuildGradle(config, (config) => {
    if (!config.modResults.contents.includes('proguard-rules-webnix.pro')) {
      config.modResults.contents = config.modResults.contents.replace(
        /proguardFiles/,
        `proguardFile "${__dirname}/proguard-rules-webnix.pro"\n            proguardFiles`
      );
    }
    return config;
  });
};
