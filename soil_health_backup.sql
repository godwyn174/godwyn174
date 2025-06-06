--
-- PostgreSQL database dump
--

-- Dumped from database version 16.9
-- Dumped by pg_dump version 16.9

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: educational_resources; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.educational_resources (
    id integer NOT NULL,
    title character varying(100) NOT NULL,
    content text NOT NULL,
    category character varying(50),
    created_at timestamp without time zone
);


ALTER TABLE public.educational_resources OWNER TO postgres;

--
-- Name: educational_resources_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.educational_resources_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.educational_resources_id_seq OWNER TO postgres;

--
-- Name: educational_resources_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.educational_resources_id_seq OWNED BY public.educational_resources.id;


--
-- Name: soil_analyses; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.soil_analyses (
    id integer NOT NULL,
    user_id integer NOT NULL,
    nitrogen double precision,
    phosphorus double precision,
    potassium double precision,
    ph_level double precision,
    organic_matter double precision,
    analysis_date timestamp without time zone,
    recommendations text
);


ALTER TABLE public.soil_analyses OWNER TO postgres;

--
-- Name: soil_analyses_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.soil_analyses_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.soil_analyses_id_seq OWNER TO postgres;

--
-- Name: soil_analyses_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.soil_analyses_id_seq OWNED BY public.soil_analyses.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(64) NOT NULL,
    password_hash character varying(128) NOT NULL,
    role character varying(20) NOT NULL,
    created_at timestamp without time zone
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: educational_resources id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.educational_resources ALTER COLUMN id SET DEFAULT nextval('public.educational_resources_id_seq'::regclass);


--
-- Name: soil_analyses id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.soil_analyses ALTER COLUMN id SET DEFAULT nextval('public.soil_analyses_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Data for Name: educational_resources; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.educational_resources (id, title, content, category, created_at) FROM stdin;
\.


--
-- Data for Name: soil_analyses; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.soil_analyses (id, user_id, nitrogen, phosphorus, potassium, ph_level, organic_matter, analysis_date, recommendations) FROM stdin;
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, username, password_hash, role, created_at) FROM stdin;
\.


--
-- Name: educational_resources_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.educational_resources_id_seq', 1, false);


--
-- Name: soil_analyses_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.soil_analyses_id_seq', 1, false);


--
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.users_id_seq', 1, false);


--
-- Name: educational_resources educational_resources_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.educational_resources
    ADD CONSTRAINT educational_resources_pkey PRIMARY KEY (id);


--
-- Name: soil_analyses soil_analyses_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.soil_analyses
    ADD CONSTRAINT soil_analyses_pkey PRIMARY KEY (id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: soil_analyses soil_analyses_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.soil_analyses
    ADD CONSTRAINT soil_analyses_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- PostgreSQL database dump complete
--

