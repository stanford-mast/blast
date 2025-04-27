# BLAST Web Frontend

## Development

To run the development server:

```bash
npm install
npm run dev
```

The web frontend will be available at http://localhost:3000

## Architecture

The web frontend is built with:
- Next.js - React framework
- TypeScript - Type safety
- Tailwind CSS - Styling
- Axios - HTTP client

It communicates with the BLAST engine server.

## Note

The web frontend is optional. If Node.js/npm are not installed, BLAST will automatically fall back to the CLI frontend. To use the web frontend, run `blastai serve`. The CLI frontend will still be available by running `blastai serve cli`.