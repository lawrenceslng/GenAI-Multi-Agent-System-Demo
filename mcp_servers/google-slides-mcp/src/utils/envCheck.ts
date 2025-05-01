import yargs from 'yargs/yargs';

export const checkEnvironmentVariables = (): void => {
  const argv = yargs(process.argv.slice(2))
    .option('google-client-id', { type: 'string' })
    .option('google-client-secret', { type: 'string' })
    .option('google-refresh-token', { type: 'string' })
    .parseSync();

  const CLIENT_ID = process.env.GOOGLE_CLIENT_ID || argv.googleClientId;
  const CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || argv.googleClientSecret;
  const REFRESH_TOKEN = process.env.GOOGLE_REFRESH_TOKEN || argv.googleRefreshToken;

  if (!CLIENT_ID || !CLIENT_SECRET || !REFRESH_TOKEN) {
    console.error('Error: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN environment variables are required.');
    console.error('Please refer to the README.md for instructions on obtaining these credentials.');
    process.exit(1);
  }
};