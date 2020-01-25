
CREATE TABLE Affiliations(
    AffiliationId BIGINT PRIMARY KEY,
    Rank INTEGER,
    NormalizedName VARCHAR(200),
    DisplayName VARCHAR(200),
    GridId VARCHAR(20),
    OfficialPage VARCHAR(255),
    WikiPage VARCHAR(200),
    PaperCount BIGINT,
    CitationCount BIGINT,
    CreatedDate DATE
  );

CREATE TABLE Authors(
    AuthorId BIGINT PRIMARY KEY,
    Rank INTEGER,
    NormalizedName VARCHAR(255),
    DisplayName VARCHAR(255),
    LastKnownAffiliationId BIGINT,
    PaperCount BIGINT,
    CitationCount BIGINT,
    CreatedDate DATE
  );


CREATE TABLE ConferenceInstances(
    ConferenceInstanceId BIGINT PRIMARY KEY,
    NormalizedName VARCHAR(100),
    DisplayName VARCHAR(100),
    ConferenceSeriesId BIGINT,
    Location VARCHAR(200),
    OfficialUrl VARCHAR(255),
    StartDate DATE,
    EndDate DATE,
    AbstractRegistrationDate DATE,
    SubmissionDeadlineDate DATE,
    NotificationDueDate DATE,
    FinalVersionDueDate DATE,
    PaperCount INT,
    CitationCount INT,
    CreatedDate DATE
    );


CREATE TABLE ConferenceSeries(
    ConferenceSeriesId BIGINT PRIMARY KEY,
    Rank INTEGER,
    NormalizedName VARCHAR(255),
    DisplayName VARCHAR(255),
    PaperCount INT,
    CitationCount INT,
    CreatedDate DATE
    );


CREATE TABLE FieldOfStudyChildren(
    FieldOfStudyId BIGINT,
    ChildFieldOfStudyId BIGINT,
    PRIMARY KEY(FieldOfStudyId,ChildFieldOfStudyId)
    );


CREATE TABLE FieldsOfStudy(
    FieldOfStudyId BIGINT PRIMARY KEY,
    Rank INTEGER,
    NormalizedName VARCHAR(255),
    DisplayName VARCHAR(255),
    MainType VARCHAR(100),
    Level INTEGER,
    PaperCount BIGINT,
    CitationCount BIGINT,
    CreatedDate DATE
    );


CREATE TABLE Journals( 
    JournalId BIGINT PRIMARY KEY,
    Rank INTEGER,
    NormalizedName VARCHAR(255),
    DisplayName VARCHAR(255),
    Issn VARCHAR(20),
    Publisher VARCHAR(50),
    Webpage VARCHAR(255),
    PaperCount BIGINT,
    CitationCount BIGINT,
    CreatedDate DATE
    );


CREATE TABLE PaperAbstracts(
    PaperId BIGINT PRIMARY KEY,
    Abstract TEXT
    );

CREATE TABLE PaperAuthorAffiliations(
    PaperId BIGINT NOT NULL,
    AuthorId BIGINT NOT NULL,
    AffiliationId BIGINT,
    AuthorSequenceNumber SMALLINT,
    OriginalAffiliation TEXT
    /** Does not work, since there can be multiple rows: PRIMARY KEY(PaperId, AuthorId,AffiliationId) **/
    );


CREATE TABLE PaperCitationContexts(
    PaperId BIGINT,
    PaperReferenceId BIGINT,
    CitationContext TEXT
    /** Does not work, since there can be multiple rows (citation contexts per paper): PRIMARY KEY(PaperId,PaperReferenceId) **/
/** last line is 31GB in size, i.e. remove **/
    );


CREATE TABLE PaperFieldsOfStudy(
    PaperId BIGINT,
    FieldOfStudyId BIGINT,
    Score REAL,
    PRIMARY KEY(PaperId,FieldOfStudyId)
    );


CREATE TABLE PaperLanguages(  
    PaperId BIGINT,
    LanguageCode VARCHAR(10)
/** PRIMARY KEY: PaperID not sufficient **/
    );



CREATE TABLE PaperReferences( 
    PaperId BIGINT,
    PaperReferenceId BIGINT,
    PRIMARY KEY(PaperId,PaperReferenceId)
    );


CREATE TABLE PaperUrls(
    PaperId BIGINT,
    SourceType INTEGER,
    SourceUrl VARCHAR(2200) PRIMARY KEY
    /** There can be several URLs for each paper, so use URL as primary key **/
    );


CREATE TABLE Papers(
    PaperId BIGINT PRIMARY KEY,
    Rank INT,
    Doi VARCHAR(200),
    DocType VARCHAR(50),
    PaperTitle TEXT,
    OriginalTitle TEXT,
    BookTitle VARCHAR(800),
    PublishedYear INTEGER,
    PublishedDate DATE,
    Publisher TEXT,
    JournalId BIGINT,
    ConferenceSeriesId BIGINT,
    ConferenceInstanceId BIGINT,
    Volume BIGINT,
    Issue BIGINT,
    FirstPage BIGINT,
    LastPage BIGINT,
    ReferenceCount BIGINT,
    CitationCount BIGINT,
    EstimatedCitation BIGINT,
    OriginalVenue TEXT,
    CreatedDate DATE
    );


CREATE TABLE RelatedFieldOfStudy(
    FieldOfStudyId1 BIGINT,
    DisplayName1 VARCHAR(100),
    Type1 VARCHAR(50),
    FieldOfStudyId2 BIGINT,
    DisplayName2 VARCHAR(100),
    Type2 VARCHAR(50),
    Rank DOUBLE PRECISION,
    PRIMARY KEY(FieldOfStudyId1, FieldOfStudyId2)
/**  PRIMARY KEY (FieldOfStudyId1, FieldOfStudyId2) not sufficient, so no PRIMARY KEY HERE, ALTERNATIVE WOULD BE PRIMARY KEY(
Field of study ID 1
Display name 1 
Field of study 2
Display name 2 
)
 **/
    );

