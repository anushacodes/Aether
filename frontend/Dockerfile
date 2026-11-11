FROM node:22-bookworm-slim@sha256:5ec23b7c622b00547753bc6ffd78085dd868a68aeb41c8e0a1418f033e13d96d AS base
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

FROM base AS dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM base AS builder
COPY --from=dependencies /app/node_modules ./node_modules
COPY frontend/package.json frontend/package-lock.json frontend/tsconfig.json frontend/next.config.ts ./
COPY frontend/app ./app
# This public URL is embedded into the browser bundle at build time.
ARG NEXT_PUBLIC_RAG_API_URL=http://localhost:18000
ENV NEXT_PUBLIC_RAG_API_URL=$NEXT_PUBLIC_RAG_API_URL
RUN npm run build

FROM base AS runtime
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static
USER node
EXPOSE 3000
CMD ["node", "server.js"]
