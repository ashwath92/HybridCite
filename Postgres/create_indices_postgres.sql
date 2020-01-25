CREATE INDEX idx_Affiliations ON Affiliations(AffiliationId);
CREATE INDEX idx_Authors ON Authors(AuthorId);
CREATE INDEX idx_ConferenceInstances ON ConferenceInstances(ConferenceInstanceId);
CREATE INDEX idx_ConferenceSeries ON ConferenceSeries(ConferenceSeriesId);
CREATE INDEX idx_FieldOfStudyChildren ON FieldOfStudyChildren(FieldOfStudyId);
CREATE INDEX idx_FieldsOfStudy ON FieldsOfStudy(FieldOfStudyId);
CREATE INDEX idx_Journals ON Journals(JournalId);
CREATE INDEX idx_PaperAbstracts ON PaperAbstracts(PaperId);
CREATE INDEX idx_PaperAuthorAffiliationsByPaper ON PaperAuthorAffiliations(PaperId);
CREATE INDEX idx_PaperAuthorAffiliationsByAuthor ON PaperAuthorAffiliations(AuthorId);
CREATE INDEX idx_PaperCitationContextsBySource ON PaperCitationContexts(PaperId);
CREATE INDEX idx_PaperCitationContextsByReference ON PaperCitationContexts(PaperReferenceId);
CREATE INDEX idx_PaperFieldsOfStudyByField ON PaperFieldsOfStudy(FieldOfStudyId);
CREATE INDEX idx_PaperFieldsOfStudyByPaper ON PaperFieldsOfStudy(PaperId);
CREATE INDEX idx_PaperLanguages ON PaperLanguages(PaperId);
CREATE INDEX idx_PaperReferencesBySource ON PaperReferences(PaperId);
CREATE INDEX idx_PaperReferencesByTarget ON PaperReferences(PaperReferenceId);
CREATE INDEX idx_PaperUrls ON PaperUrls(PaperId);
CREATE INDEX idx_Papers ON Papers(PaperId);
CREATE INDEX idx_RelatedFieldOfStudy1 ON Papers(FieldOfStudyId1);
CREATE INDEX idx_RelatedFieldOfStudy2 ON Papers(FieldOfStudyId2);

CREATE INDEX idx_AuthorNames ON Authors(DisplayName);
/** idx_ConferenceInstanceName may not be necessary**/
CREATE INDEX idx_ConferenceInstanceName ON ConferenceInstances(DisplayName);
CREATE INDEX idx_FieldsOfStudyName ON FieldsOfStudy(DisplayName);
CREATE INDEX idx_PapersTitle ON Papers(OriginalTitle);
CREATE INDEX idx_BookTitle ON Papers(booktitle);