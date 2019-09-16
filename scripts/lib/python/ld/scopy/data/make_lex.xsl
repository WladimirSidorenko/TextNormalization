<?xml version="1.0" encoding="UTF-8" ?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" indent="yes" />
    <xsl:template match="text()"/>
    <xsl:template match="/">
        <xsl:element name="connlex">
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>
    <xsl:template match="eintrag">
        <xsl:element name="entry">
            <xsl:attribute name="id">
                <xsl:number/>
            </xsl:attribute>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>
    <xsl:template match="orth">
        <xsl:copy-of select="."/>
    </xsl:template>
    <xsl:template match="syn">
        <xsl:element name="syn">
            <xsl:choose>
                <xsl:when test="kat/text()='konj'">
                    <xsl:attribute name="type">conj</xsl:attribute>
                </xsl:when>
                <xsl:when test="kat/text()='coordconj'">
                    <xsl:attribute name="type">conj</xsl:attribute>
                </xsl:when>
                <xsl:when test="kat/text()='subj'">
                    <xsl:attribute name="type">subj</xsl:attribute>
                </xsl:when>
                <xsl:when test="kat/text()='postp'">
                    <xsl:attribute name="type">subj</xsl:attribute>
                </xsl:when>
                <xsl:when test="kat/text()='praep'">
                    <xsl:attribute name="type">prep</xsl:attribute>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:attribute name="type">adv</xsl:attribute>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:element name="order">
                <xsl:copy-of select="abfolge/*"/>
            </xsl:element>
            <xsl:for-each select="sem/coherence_relations">
                <xsl:if test="relation">
                    <xsl:element name="rel">
                        <xsl:value-of select="relation"/>
                    </xsl:element>
                </xsl:if>
            </xsl:for-each>
        </xsl:element>
    </xsl:template>
</xsl:stylesheet> 
