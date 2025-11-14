# syntax=docker/dockerfile:1.6

ARG TARGETPLATFORM=linux/amd64
ARG VITE_API_URL

FROM node:20-alpine AS build
WORKDIR /app
ENV VITE_API_URL=${VITE_API_URL}
COPY package.json package-lock.json* pnpm-lock.yaml* bun.lockb* ./
RUN if [ -f package-lock.json ]; then npm ci; \
    elif [ -f pnpm-lock.yaml ]; then npm install -g pnpm && pnpm install --frozen-lockfile; \
    elif [ -f bun.lockb ]; then npm install -g bun && bun install; \
    else npm install; fi
COPY . .
RUN npm run build

FROM --platform=${TARGETPLATFORM} nginx:1.27-alpine
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}
COPY --from=build /app/dist /usr/share/nginx/html
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
